"""
Interactive Phi-4-mini Terminal
===============================

Port of run_qwen.py to Phi-4-mini-instruct. Same manual autoregressive loop, same
ipynb-style embedding perturbation interface, so the sweep notebook's run_one
monkeypatching works unchanged.

  Model : microsoft/Phi-4-mini-instruct  (Phi3ForCausalLM, dense, 3.8B, ~7.6 GB bf16)
  MIT licensed and UNGATED. Text-only. No thinking mode.

THE ONE THING THAT BREAKS YOUR HARNESS
--------------------------------------
Phi-4-mini has NO q_proj. Verified against the real Phi3ForCausalLM: each attention
block exposes exactly two children, o_proj and qkv_proj. The query, key and value
projections are FUSED into one nn.Linear whose output is [q | k | v] concatenated along
the last dim, width (num_q_heads + 2*num_kv_heads) * head_dim.

This matters more than a normal incompatibility because it fails SILENTLY. The
name-based discovery used for every other model in this series looks for modules ending
in "q_proj", finds zero, registers zero hooks, and the sweep runs to completion with an
empty _captured_query_vectors dict. You would get 12,000 generations and no query data,
and nothing would raise. Hooks here attach to qkv_proj and slice the first
num_q_heads * head_dim columns, so downstream code sees query vectors exactly as before.

Run verify_qkv_split(model) after loading. It checks the fused width against the config
arithmetic and confirms the slice boundary, which is the one place a silent off-by-a-head
would corrupt every query vector without any error.

Other differences from Qwen2.5, all minor:

  - SHARED INPUT/OUTPUT EMBEDDINGS. Phi-4-mini ties embed_tokens to the LM head. The
    perturbation is unaffected (noise goes on activations, not weights), but it is worth
    a line in the paper: the noise lives in the same space the model unembeds through,
    which is not true of the untied models in your set.
  - 200K VOCABULARY, larger than Qwen2.5's 152K, so embedding-scale stats differ. Run
    embedding_stats() as usual.
  - PARTIAL ROPE. About 25% of each head dim is left position-agnostic. Nothing to do,
    but it means query vectors are not directly comparable to a full-RoPE model's.
  - trust_remote_code. Microsoft's card passes trust_remote_code=True. Recent
    transformers has native phi3 support, so load_model tries native first and only
    falls back to remote code if that fails, with a printed notice either way.
  - NO THINKING MODE. Phi-4-mini-instruct answers directly. (Phi-4-mini-reasoning is a
    separate checkpoint and would need the think-splitting the Qwen3/Gemma ports have.)

SAMPLING NOTE FOR THE PAPER
   Kept at your cross-model 0.7 / 50 / 0.9.

Usage:
  python run_phi4mini.py

Environment overrides:
  PHI4_MODEL_ID            default microsoft/Phi-4-mini-instruct
  PHI4_DEVICE              cuda | dml | cpu   (QWEN_DEVICE also honoured)
  PHI4_SKIP_EMBED_PROMPT   1 to skip the inspect/perturb prompts
  PHI4_TRUST_REMOTE_CODE   1 to force the remote-code path
  HF_TOKEN                 optional; MIT and not gated

Commands inside the terminal:
  /verbose /temp /topk /topp /max /eosonly /system /clear /layers /help /quit
  /premebed /embedfile /embstats /qkv
"""

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    os.system("")

import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime

try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False


def _env(*names, default=""):
    """First non-empty value among the given env var names. PHI4_* wins over QWEN_*."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip() != "":
            return v
    return default


def _cuda_dtype():
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _select_device():
    global DEVICE, DTYPE
    forced = _env("PHI4_DEVICE", "QWEN_DEVICE").strip().lower()

    if forced == "cpu":
        DEVICE, DTYPE = "cpu", torch.float32
        print("  [Device] PHI4_DEVICE=cpu, using CPU.\n")
        return

    if forced == "cuda":
        if torch.cuda.is_available():
            DEVICE, DTYPE = "cuda", _cuda_dtype()
            print(f"  [Device] PHI4_DEVICE=cuda, using {torch.cuda.get_device_name(0)}.\n")
            return
        print("  [Device] PHI4_DEVICE=cuda but CUDA is not available; falling back.\n")

    if forced in ("dml", "directml"):
        if _HAS_DML:
            try:
                DEVICE = torch_directml.device()
                DTYPE = torch.float16
                print(f"  [Device] PHI4_DEVICE=dml, DirectML: {torch_directml.device_name(0)}.\n")
                return
            except Exception as e:
                print(f"  [Device] DirectML requested but failed ({e}); falling back.\n")
        else:
            print("  [Device] PHI4_DEVICE=dml but torch_directml is not installed; falling back.\n")

    if torch.cuda.is_available():
        DEVICE, DTYPE = "cuda", _cuda_dtype()
        print(f"  [Device] CUDA, {torch.cuda.get_device_name(0)}\n")
        return

    if _HAS_DML:
        try:
            DEVICE = torch_directml.device()
            DTYPE = torch.float16
            print(f"  [Device] DirectML, {torch_directml.device_name(0)}\n")
            return
        except Exception as e:
            print(f"  [Device] DirectML import OK but init failed: {e}\n")

    DEVICE, DTYPE = "cpu", torch.float32
    print("  [Device] CPU (no CUDA GPU and no DirectML).")
    print("  NOTE: at 3.8B this is actually survivable on CPU, unlike the 8B+ models,")
    print("        but still far too slow for a 3000-generation sweep.\n")


# ─── Hook storage ────────────────────────────────────────────────────────────
_captured_query_vectors = {}
_pre_layer_embed_runtime = False
_write_phi4_embed_log_file = False
_current_generation_step = 0
_ATTN_LAYER_IDS = []
_Q_WIDTH = None      # num_q_heads * head_dim, the slice boundary in the fused qkv
_FUSED_QKV = False

try:
    _PHI4_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _PHI4_SCRIPT_DIR = os.getcwd()
PHI4_OUTPUT_LOG_PATH = os.path.join(_PHI4_SCRIPT_DIR, "Phi4mini output log.txt")
WRITE_PHI4_EMBED_LOG_FILE = False


def _append_phi4_output_log(tokenizer, generation_step, input_ids, emb):
    """Append input tokens, per-token embedding rows, and the full matrix to disk."""
    ids_flat = input_ids[0].detach().cpu().tolist()
    seq_len = len(ids_flat)
    hidden = emb.shape[-1]
    mat = emb[0].detach().float().cpu()

    with open(PHI4_OUTPUT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"{datetime.now().isoformat()}  generation_step={generation_step}\n")
        f.write(
            f"input_ids shape [1, {seq_len}]  "
            f"embedding output (into layers) shape [seq_len={seq_len}, hidden={hidden}]\n"
        )
        f.write("(source: model.model.embed_tokens, initial hidden_states for the decoder)\n")

        f.write("\n--- Input tokens ---\n")
        f.write(f"{'idx':<8}{'tok_id':<12}decoded\n")
        for i, tid in enumerate(ids_flat):
            f.write(f"{i:<8}{tid:<12}{repr(tokenizer.decode([tid]))}\n")

        f.write("\n--- Per-token embedding vectors (full hidden_size per row) ---\n")
        for i, tid in enumerate(ids_flat):
            f.write(f"\ntoken_idx={i} tok_id={tid} decoded={repr(tokenizer.decode([tid]))}\n")
            f.write(" ".join(f"{x:.8g}" for x in mat[i].tolist()) + "\n")

        f.write("\n--- Final matrix into layers [seq_len x hidden] ---\n")
        for i in range(seq_len):
            f.write(" ".join(f"{x:.8g}" for x in mat[i].tolist()) + "\n")


def _make_pre_layer_embed_hook(tokenizer):
    def hook_fn(module, inputs, output):
        if not (_pre_layer_embed_runtime or _write_phi4_embed_log_file):
            return
        input_ids = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        emb = output
        if _write_phi4_embed_log_file:
            _append_phi4_output_log(tokenizer, _current_generation_step, input_ids, emb)
        if not _pre_layer_embed_runtime:
            return
        ids_flat = input_ids[0].detach().cpu().tolist()
        seq_len = len(ids_flat)
        h = min(5, emb.shape[-1])
        print(f"\n── Pre-stack embedding (before attention / MLP) ─────────")
        print(f"  generation_step={_current_generation_step}  seq_len={seq_len}")
        dim_hdr = "".join(f"{('dim' + str(i)):>12}" for i in range(h))
        print(f"  {'idx':<5}{'tok_id':<10}{'decoded':<22}{dim_hdr}")

        def row_str(i):
            tid = ids_flat[i]
            dec = repr(tokenizer.decode([tid]))[:20]
            vec = emb[0, i, :h].detach().float().cpu().tolist()
            return f"  {i:<5}{tid:<10}{dec:<22}" + "".join(f"{x:>12.5f}" for x in vec)

        if seq_len <= PRE_LAYER_EMBED_MAX_ROWS:
            for i in range(seq_len):
                print(row_str(i))
        else:
            half = PRE_LAYER_EMBED_MAX_ROWS // 2
            for i in range(half):
                print(row_str(i))
            print(f"  ... ({seq_len - 2 * half} rows omitted) ...")
            for i in range(seq_len - half, seq_len):
                print(row_str(i))
        print()

    return hook_fn


def _make_attn_hook(layer_idx, q_width=None):
    """Capture query vectors.

    On Phi-4-mini the hooked module is the FUSED qkv_proj, so slice off the query part:
    output[..., :q_width] where q_width = num_q_heads * head_dim. Without the slice the
    captured tensor silently includes the key and value blocks and every query-vector
    number in the analysis is wrong.
    """
    def hook_fn(module, input, output):
        q = output if q_width is None else output[..., :q_width]
        _captured_query_vectors[layer_idx] = q.detach()
    return hook_fn


def _find_attn_projections(model):
    """Locate per-layer query projections, fused or separate.

    Returns (mapping, fused_flag). Prefers a real q_proj when one exists so this file
    still works if pointed at a non-Phi checkpoint via PHI4_MODEL_ID.
    """
    sep, fused = {}, {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        if name.endswith("q_proj"):
            sep[int(m.group(1))] = (name, mod)
        elif name.endswith("qkv_proj"):
            fused[int(m.group(1))] = (name, mod)
    if sep:
        return sep, False
    return fused, True


def _head_dim(model):
    cfg = model.config
    hd = getattr(cfg, "head_dim", None)
    if hd:
        return int(hd)
    return int(cfg.hidden_size // cfg.num_attention_heads)


def verify_qkv_split(model, verbose=True):
    """Confirm the fused qkv layout matches the config arithmetic.

    The slice boundary is the single point where a silent error would corrupt every
    query vector without raising anything, so check it explicitly rather than trusting
    that q comes first and that head_dim is what the config implies.
    """
    cfg = model.config
    hd = _head_dim(model)
    nq = int(cfg.num_attention_heads)
    nkv = int(getattr(cfg, "num_key_value_heads", nq))
    expected = (nq + 2 * nkv) * hd
    q_width = nq * hd

    mapping, fused = _find_attn_projections(model)
    if not mapping:
        raise RuntimeError("no q_proj or qkv_proj found, cannot hook query vectors")

    idx, (name, mod) = sorted(mapping.items())[0]
    actual = mod.weight.shape[0]

    if verbose:
        print("\n── Attention projection layout ────────────────────────────")
        print(f"  module         : {name}")
        print(f"  fused qkv      : {fused}")
        print(f"  q heads        : {nq}   kv heads: {nkv}   head_dim: {hd}")
        print(f"  out width      : {actual}  (config predicts {expected})")
        print(f"  query slice    : [..., :{q_width}]")

    if fused:
        if actual != expected:
            raise RuntimeError(
                f"fused qkv width {actual} != predicted {expected}. The [q|k|v] split is "
                f"not what the config implies; do NOT trust the query slice."
            )
        if verbose:
            print(f"  layout verified: q occupies columns 0..{q_width-1}, "
                  f"k {q_width}..{q_width + nkv*hd - 1}, v the rest")
    elif verbose:
        print("  separate q_proj, no slicing needed")

    if verbose:
        print(f"  layers with a query projection: {len(mapping)} of "
              f"{cfg.num_hidden_layers}\n")

    if len(mapping) != cfg.num_hidden_layers:
        raise RuntimeError(
            f"found query projections on {len(mapping)} of {cfg.num_hidden_layers} "
            f"layers, hooks would silently miss the rest"
        )
    return {"fused": fused, "q_width": q_width, "head_dim": hd,
            "n_layers": len(mapping), "out_width": actual}


# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME     = _env("PHI4_MODEL_ID", default="microsoft/Phi-4-mini-instruct")
MAX_NEW_TOKENS = 2048
REQUIRE_EOS_BEFORE_PRINT = False
TEMPERATURE    = 0.7
TOP_K          = 50
TOP_P          = 0.9
VERBOSE        = False
PRINT_PRE_LAYER_EMBED = False
PRE_LAYER_EMBED_MAX_ROWS = 48
EMBED_INSPECT_MAX_ROWS = 64
CTX_SAFETY_CAP = 8192

DEVICE = "cpu"
DTYPE = torch.float32
_select_device()

SYSTEM_PROMPT  = ""
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """Download (first time) and load the model + tokenizer."""
    import transformers

    print(f"\n{'='*60}")
    print(f"  Loading model: {MODEL_NAME}")
    print(f"{'='*60}\n")
    print(f"  Device : {DEVICE}")
    print(f"  Dtype  : {DTYPE}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  (First run downloads ~7.6 GB of weights.)\n")

    hf_token = _env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", default=None) or None
    force_remote = _env("PHI4_TRUST_REMOTE_CODE", default="0").strip() == "1"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = {"low_cpu_mem_usage": True}
    if hf_token:
        kw["token"] = hf_token

    model = None
    if not force_remote:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, dtype=DTYPE, **kw
            ).to(DEVICE)
            print("  Loaded with native transformers phi3 support (no remote code).")
        except Exception as e:
            print(f"  Native load failed ({type(e).__name__}); retrying with "
                  f"trust_remote_code=True.")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=DTYPE, trust_remote_code=True, **kw
        ).to(DEVICE)
        print("  Loaded with trust_remote_code=True.")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model loaded!")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Vocabulary size: {len(tokenizer):,}")
    print(f"  Max positions  : {model.config.max_position_embeddings:,}")
    print(f"  Hidden size    : {model.config.hidden_size}")
    print(f"  Num layers     : {model.config.num_hidden_layers}")
    print(f"  Num attn heads : {model.config.num_attention_heads}")
    print(f"  Num KV heads   : {getattr(model.config, 'num_key_value_heads', 'n/a')}")
    print(f"  Head dimension : {_head_dim(model)}")
    tied = getattr(model.config, "tie_word_embeddings", None)
    print(f"  Tied embeddings: {tied}"
          f"{'  (input table is also the output projection)' if tied else ''}")
    print(f"  Stop token ids : {sorted(get_stop_token_ids(tokenizer))}")

    # ── Query-vector hooks. Phi-4-mini has NO q_proj; slice the fused qkv. ──
    global _ATTN_LAYER_IDS, _Q_WIDTH, _FUSED_QKV
    info = verify_qkv_split(model, verbose=True)
    _Q_WIDTH = info["q_width"] if info["fused"] else None
    _FUSED_QKV = info["fused"]

    mapping, _ = _find_attn_projections(model)
    for idx, (mname, mmod) in sorted(mapping.items()):
        mmod.register_forward_hook(_make_attn_hook(idx, q_width=_Q_WIDTH))
    _ATTN_LAYER_IDS = sorted(mapping.keys())
    print(f"  Registered query-vector hooks on {len(mapping)} layers"
          f"{' (fused qkv, sliced to the q block)' if _FUSED_QKV else ''}.")

    emb_mod = model.get_input_embeddings()
    emb_mod.register_forward_hook(_make_pre_layer_embed_hook(tokenizer))
    print(f"  Registered pre-stack embedding hook on {type(emb_mod).__name__}.\n")

    return model, tokenizer


def build_prompt(tokenizer, messages):
    """Apply the chat template.

    Phi-4-mini uses <|system|> / <|user|> / <|assistant|> turns terminated by <|end|>.
    No thinking mode and no template kwargs to worry about, so this is the simplest
    build_prompt in the series.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def encode_prompt(tokenizer, prompt_text):
    """Tokenize the chat-template string without adding extra specials."""
    return tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)


def get_stop_token_ids(tokenizer):
    """EOS-family stop tokens for Phi-4-mini.

    <|end|> terminates a turn; <|endoftext|> is the base EOS. As elsewhere, do NOT union
    additional_special_tokens_ids, the role markers in there would truncate replies to
    empty under noisy logits.
    """
    stop = set()
    eos = tokenizer.eos_token_id
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            stop.update(int(e) for e in eos)
        else:
            stop.add(int(eos))
    unk = getattr(tokenizer, "unk_token_id", None)
    for name in ("<|end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
        except Exception:
            continue
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            stop.add(tid)
    return stop


def show_token_table(tokenizer, token_ids):
    print(f"  {'Index':<8}{'Token ID':<12}{'Decoded':<30}")
    print(f"  {'─'*8}{'─'*12}{'─'*30}")
    for i, tid in enumerate(token_ids):
        print(f"  {i:<8}{tid:<12}{repr(tokenizer.decode([tid])):<30}")


def show_model_layers(model):
    print("\n── Model Layer Structure ──────────────────────────────────")
    for name, param in model.named_parameters():
        print(f"  {name:<60} {str(list(param.shape)):<25} {param.dtype}")
    print()


def embedding_stats(model, sample=8192):
    """Scale reference, measured on the module OUTPUT for consistency with the other
    ports (Gemma scales inside embed_tokens; Phi does not, so here output == weight
    lookup, but keep the method identical so the numbers are comparable).
    """
    emb = model.get_input_embeddings()
    W = emb.weight
    vocab = int(W.shape[0])
    n = min(sample, vocab)
    idx = torch.randperm(vocab, device=W.device)[:n].unsqueeze(0)
    with torch.no_grad():
        out = emb(idx)[0].detach().float()

    stats = {
        "vocab": vocab,
        "hidden": int(W.shape[1]),
        "elem_std": float(out.std()),
        "elem_absmean": float(out.abs().mean()),
        "row_l2_mean": float(out.norm(dim=-1).mean()),
        "row_rms_mean": float(out.pow(2).mean(dim=-1).sqrt().mean()),
    }
    print("\n── Input embedding scale (measured on module OUTPUT) ──────")
    for k, v in stats.items():
        print(f"  {k:<14}: {v:,.6g}" if isinstance(v, float) else f"  {k:<14}: {v:,}")
    print("  (sigma / row_rms_mean = perturbation strength relative to a token vector)\n")
    return stats


@torch.inference_mode()
def verify_cache_consistency(model, tokenizer, prompt="Explain gravity in one sentence.",
                             n_tokens=24):
    """Check the manual step loop against model.generate() under greedy decoding."""
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    text = build_prompt(tokenizer, msgs)
    ids = encode_prompt(tokenizer, text).to(DEVICE)

    ref = model.generate(
        input_ids=ids, max_new_tokens=n_tokens, do_sample=False,
        pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    ref_new = ref[0, ids.shape[1]:].tolist()

    cfg = {
        "verbose": False, "temperature": 0.0, "top_k": 0, "top_p": 1.0,
        "max_new_tokens": n_tokens, "print_pre_layer_embed": False,
        "write_phi4_embed_log_file": False, "require_eos_before_print": False,
    }
    prev = os.environ.get("PHI4_SKIP_EMBED_PROMPT", "")
    os.environ["PHI4_SKIP_EMBED_PROMPT"] = "1"
    try:
        mine = generate_response(model, tokenizer, text, cfg, _return_ids=True)
    finally:
        os.environ["PHI4_SKIP_EMBED_PROMPT"] = prev

    k = min(len(ref_new), len(mine))
    match = ref_new[:k] == mine[:k]
    print("\n── Cache consistency check (greedy) ───────────────────────")
    print(f"  generate() : {ref_new[:k]}")
    print(f"  manual loop: {mine[:k]}")
    print(f"  MATCH: {match}")
    if not match:
        first = next((i for i in range(k) if ref_new[i] != mine[i]), None)
        print(f"  First divergence at token {first}. Do NOT run the sweep until fixed.")
    print()
    return match


def _embed_inspect_row_indices(seq_len, max_rows=EMBED_INSPECT_MAX_ROWS):
    if seq_len <= max_rows:
        return list(range(seq_len)), None
    half = max_rows // 2
    return list(range(half)) + list(range(seq_len - half, seq_len)), seq_len - 2 * half


def ipynb_style_inspect_input_print(tokenizer, input_ids, embeddings):
    """Print tokens and first 5 embedding dims (LLM_Perturbation.ipynb style)."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)
    idx_rows, omitted = _embed_inspect_row_indices(seq_len)
    half = len(idx_rows) // 2 if omitted is not None else None

    print("\nTOKENS")
    for j, i in enumerate(idx_rows):
        if omitted is not None and j == half:
            print(f"... ({omitted} rows omitted) ...")
        print(f"{i:02d} | {tokens[i]}")

    print("\nEMBEDDING VECTORS (first 5 dims)")
    for j, i in enumerate(idx_rows):
        if omitted is not None and j == half:
            print(f"... ({omitted} rows omitted) ...")
        vec = embeddings[0, i, :5].detach().float().cpu().tolist()
        print(f"{tokens[i]:>12} : {vec}")


def ipynb_style_maybe_modify_embeddings(embeddings):
    """Interactive Y/N and noise scale; same rule as the notebook.

    The sweep notebook monkeypatches this with a fixed-sigma version. generate_response
    resolves the name as a module global at call time, so patching
    run_phi4mini.ipynb_style_maybe_modify_embeddings works without editing this file.
    """
    choice = input("\nModify embeddings? (Y/N): ").strip().upper()
    if choice != "Y":
        print("Proceeding without modification.")
        return 0.0, None

    noise_scale = float(input("Enter noise scale (e.g. 0.5, 0.7, 1.0): ").strip())
    print(f"Applying NON-uniform perturbation with scale {noise_scale}")
    return noise_scale, embeddings + noise_scale * torch.randn_like(embeddings)


def ipynb_style_inspect_layers_print(tokenizer, outputs, top_k=5):
    """Per-layer last-token hidden (first 5 dims) and top-k next-token probs."""
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        print("\n(inspect layers: hidden_states not available)\n")
        return

    print("\nLAYER-WISE HIDDEN STATES (last token, first 5 dims)")
    for i, layer in enumerate(hidden_states):
        vec = layer[0, -1, :5].detach().float().cpu().tolist()
        print(f"Layer {i:02d}: {vec}")

    logits = outputs.logits[0, -1]
    probs = F.softmax(logits.float(), dim=-1)
    k = min(top_k, probs.numel())
    tk = torch.topk(probs, k)

    print(f"\nTOP-{k} NEXT TOKEN PROBABILITIES")
    for idx, p in zip(tk.indices, tk.values):
        print(f"{repr(tokenizer.decode([idx.item()]))} : {p.item():.4f}")


@torch.inference_mode()
def generate_response(model, tokenizer, prompt_text, config, _return_ids=False):
    """
    Manual autoregressive generation loop.

    Perturbation semantics, identical across the series: noise is applied once, to the
    prompt embeddings, at the prefill step. Every generated token afterwards is embedded
    normally and attends to the perturbed prefix through the KV cache.

    As in the Gemma port, prefill always goes through inputs_embeds regardless of sigma,
    so sigma=0 is a true control rather than a different code path.
    """
    global _pre_layer_embed_runtime, _write_phi4_embed_log_file, _current_generation_step

    verbose       = config["verbose"]
    temperature   = config["temperature"]
    top_k         = config["top_k"]
    top_p         = config["top_p"]
    max_new       = config["max_new_tokens"]
    require_eos   = config.get("require_eos_before_print", REQUIRE_EOS_BEFORE_PRINT)
    _pre_layer_embed_runtime = config.get("print_pre_layer_embed", PRINT_PRE_LAYER_EMBED)
    _write_phi4_embed_log_file = config.get(
        "write_phi4_embed_log_file",
        config.get("write_qwen_embed_log_file", WRITE_PHI4_EMBED_LOG_FILE),
    )

    input_ids = encode_prompt(tokenizer, prompt_text).to(DEVICE)
    num_prompt_tokens = input_ids.shape[1]
    ctx_limit = min(model.config.max_position_embeddings, CTX_SAFETY_CAP)
    max_new = min(max_new, max(1, ctx_limit - num_prompt_tokens - 1))

    if verbose:
        print(f"\n── Tokenization ──────────────────────────────────────────")
        print(f"  Prompt length: {num_prompt_tokens} tokens")
        show_token_table(tokenizer, input_ids[0].tolist()[-20:])
        print()

    stop_token_ids = get_stop_token_ids(tokenizer)

    if verbose:
        print(f"── Autoregressive Generation ─────────────────────────────")
        print(f"  max_new_tokens={max_new}  temperature={temperature}  "
              f"top_k={top_k}  top_p={top_p}")
        print(f"  Stop token IDs: {sorted(stop_token_ids)}")
        print()

    embed_layer = model.get_input_embeddings()
    base_emb = embed_layer(input_ids)
    noise_scale = 0.0
    first_step_embeds = None
    skip_embed_prompt = _env(
        "PHI4_SKIP_EMBED_PROMPT", "QWEN_SKIP_EMBED_PROMPT"
    ).strip() == "1"
    if not skip_embed_prompt:
        ipynb_style_inspect_input_print(tokenizer, input_ids, base_emb)
        noise_scale, first_step_embeds = ipynb_style_maybe_modify_embeddings(base_emb)
    elif verbose:
        print("\n(PHI4_SKIP_EMBED_PROMPT=1, skipping embedding inspect / perturb)\n")

    if first_step_embeds is not None and first_step_embeds.dtype != base_emb.dtype:
        first_step_embeds = first_step_embeds.to(base_emb.dtype)

    generated_ids = []
    past_key_values = None
    stop_reason = "max_tokens"
    t_start = time.time()

    for step in range(max_new):
        _captured_query_vectors.clear()
        _current_generation_step = step
        output_hidden_states = verbose or (step == 0 and not skip_embed_prompt)

        # One path for every sigma: prefill on embeds, then last-token embeds.
        if past_key_values is None:
            step_embeds = (first_step_embeds
                           if (noise_scale > 0.0 and first_step_embeds is not None)
                           else base_emb)
        else:
            step_embeds = embed_layer(input_ids[:, -1:])

        outputs = model(
            inputs_embeds=step_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=output_hidden_states if past_key_values is None else verbose,
            output_attentions=False,
        )
        past_key_values = outputs.past_key_values

        if step == 0 and not skip_embed_prompt and getattr(outputs, "hidden_states", None):
            ipynb_style_inspect_layers_print(tokenizer, outputs, top_k=5)

        logits = outputs.logits[:, -1, :].float()

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        if verbose:
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            raw_logits = outputs.logits[:, -1, :].float()[0]
            print(f"  ┌─ Step {step} ────────────────────────────────────────")

            hd = _head_dim(model)
            n_layers = model.config.num_hidden_layers
            for li in sorted(set([0, n_layers // 2, n_layers - 1])):
                if li in _captured_query_vectors:
                    # already sliced to the q block by the hook
                    q_head0 = _captured_query_vectors[li][0, -1].float()[:hd]
                    vals = "".join(f"{v:>10.4f}" for v in q_head0[:5].tolist())
                    print(f"  │  L{li:<6}{vals}  || {q_head0.norm().item():.4f}")
            print(f"  │")

            print(f"  │  Top-10 candidates:")
            for rank in range(min(10, len(sorted_indices))):
                tid = sorted_indices[rank].item()
                marker = " ◄ EOS" if tid in stop_token_ids else ""
                print(f"  │  {rank+1:<6}{tid:<10}{repr(tokenizer.decode([tid])):<28}"
                      f"{sorted_probs[rank].item():<12.6f}{raw_logits[tid].item():<10.4f}{marker}")

            if getattr(outputs, "hidden_states", None):
                lh = outputs.hidden_states[-1]
                print(f"  │  Hidden state (last layer): shape={list(lh.shape)}, "
                      f"norm={lh[0, -1].float().norm().item():.4f}")

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[0, -1]] = float('-inf')

        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        filtered_probs = F.softmax(logits, dim=-1)
        if temperature == 0:
            next_token = torch.argmax(filtered_probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(
                filtered_probs.float().cpu(), num_samples=1
            ).to(DEVICE)

        next_id = next_token.item()
        generated_ids.append(next_id)

        if verbose:
            print(f"  │  ✦ SAMPLED: id={next_id}  "
                  f"token={repr(tokenizer.decode([next_id]))}  "
                  f"prob={probs[0, next_id].item():.6f}")
            print(f"  └{'─'*56}")

        if next_id in stop_token_ids:
            stop_reason = "eos"
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    elapsed = time.time() - t_start

    if _return_ids:
        return generated_ids

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if require_eos and stop_reason != "eos":
        if answer or generated_ids:
            print(
                "\n  [No reply printed] Generation stopped on max length / context cap "
                f"({stop_reason}), not on EOS. Suppressed because require_eos_before_print "
                "is ON. Use /eosonly to turn that OFF.\n"
            )
        answer = ""

    if verbose:
        print(f"\n── Generation Summary ────────────────────────────────────")
        print(f"  Tokens generated : {len(generated_ids)}")
        print(f"  Time             : {elapsed:.1f}s "
              f"({len(generated_ids)/elapsed if elapsed else 0:.1f} tok/s)")
        print(f"  Stop reason      : {stop_reason}")
        print(f"  Answer           : {repr(answer)}")
        print()

    return answer


def print_help(config):
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Commands                                                ║
╠══════════════════════════════════════════════════════════╣
║  /verbose    Toggle verbose mode  (currently: {str(config['verbose']):<10}) ║
║  /premebed   Toggle pre-layer embed print                ║
║  /embedfile  Log tokens + full embed matrix to disk      ║
║  /temp N     Set temperature      (currently: {config['temperature']:<10}) ║
║  /topk N     Set top-k            (currently: {config['top_k']:<10}) ║
║  /topp N     Set top-p            (currently: {config['top_p']:<10}) ║
║  /max N      Set max new tokens   (currently: {config['max_new_tokens']:<10}) ║
║  /eosonly    Require EOS to print                        ║
║  /qkv        Verify the fused qkv split / query slice    ║
║  /embstats   Print input-embedding scale stats           ║
║  /system X   Set system prompt                           ║
║  /clear      Clear conversation history                  ║
║  /layers     Print model layer names & shapes            ║
║  /help       Show this help                              ║
║  /quit       Exit                                        ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    model, tokenizer = load_model()

    config = {
        "verbose":                  VERBOSE,
        "temperature":              TEMPERATURE,
        "top_k":                    TOP_K,
        "top_p":                    TOP_P,
        "max_new_tokens":           MAX_NEW_TOKENS,
        "print_pre_layer_embed":    PRINT_PRE_LAYER_EMBED,
        "write_phi4_embed_log_file": WRITE_PHI4_EMBED_LOG_FILE,
        "require_eos_before_print": REQUIRE_EOS_BEFORE_PRINT,
    }

    system_prompt = SYSTEM_PROMPT
    conversation = []

    print(f"{'='*60}")
    print(f"  Phi-4-mini Interactive Terminal")
    print(f"  Type a question and press Enter.")
    print(f"  Type /help for commands, /quit to exit.")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_help(config)
            elif cmd == "/verbose":
                config["verbose"] = not config["verbose"]
                print(f"  Verbose mode: {'ON' if config['verbose'] else 'OFF'}")
            elif cmd == "/premebed":
                config["print_pre_layer_embed"] = not config.get("print_pre_layer_embed", False)
                print(f"  Pre-layer embedding debug: {config['print_pre_layer_embed']}")
            elif cmd == "/embedfile":
                config["write_phi4_embed_log_file"] = not config.get(
                    "write_phi4_embed_log_file", WRITE_PHI4_EMBED_LOG_FILE)
                print(f"  Embedding file log: {config['write_phi4_embed_log_file']} "
                      f"→ {PHI4_OUTPUT_LOG_PATH}")
            elif cmd == "/temp":
                config["temperature"] = float(arg)
                print(f"  Temperature set to {config['temperature']}")
            elif cmd == "/topk":
                config["top_k"] = int(arg)
                print(f"  Top-k set to {config['top_k']}")
            elif cmd == "/topp":
                config["top_p"] = float(arg)
                print(f"  Top-p set to {config['top_p']}")
            elif cmd == "/max":
                config["max_new_tokens"] = int(arg)
                print(f"  Max new tokens set to {config['max_new_tokens']}")
            elif cmd == "/eosonly":
                config["require_eos_before_print"] = not config.get(
                    "require_eos_before_print", REQUIRE_EOS_BEFORE_PRINT)
                print(f"  Require EOS before print: {config['require_eos_before_print']}")
            elif cmd == "/qkv":
                verify_qkv_split(model)
            elif cmd == "/embstats":
                embedding_stats(model)
            elif cmd == "/system":
                system_prompt = arg
                print(f"  System prompt set to: {repr(system_prompt)}")
            elif cmd == "/clear":
                conversation.clear()
                print("  Conversation history cleared.")
            elif cmd == "/layers":
                show_model_layers(model)
            else:
                print(f"  Unknown command: {cmd}.  Type /help for commands.")
            continue

        conversation.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": system_prompt}] + conversation
        prompt_text = build_prompt(tokenizer, messages)
        answer = generate_response(model, tokenizer, prompt_text, config)

        if answer:
            conversation.append({"role": "assistant", "content": answer})
            print(answer + "\n")


if __name__ == "__main__":
    main()
