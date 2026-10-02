"""
Interactive Mistral-7B-Instruct-v0.3 Terminal
=============================================

A real, modern, instruction-tuned LLM running locally via HuggingFace Transformers.

  Model      : mistralai/Mistral-7B-Instruct-v0.3 (7 billion parameters)
  Source code : modeling_mistral.py  (copied to this folder from transformers package)
  Inference   : This script performs a manual autoregressive generation loop so
                you can inspect every token, probability distribution, and vector.

  For the C/C++ inference engine source, see:  llama.cpp/

Usage:
  python run_mistral.py

  By default only the model reply is printed (no "Assistant:" label). Use /verbose for
  token tables and generation summary; /premebed for short embedding debug on screen;
  /embedfile to append full tokens, per-token embedding vectors, and the [seq_len, hidden]
  matrix to  Mistral output log.txt  (same folder as this script).

GPU (optional):
  - This file uses PyTorch only, it cannot use Vulkan (Vulkan is not a PyTorch backend).
  - AMD laptop (no CUDA): for Vulkan GPU acceleration use  python run_mistral_llama.py  (llama.cpp).
  - To GPU-accelerate *this* script on Windows AMD/Intel:  pip install torch-directml  (DirectX 12; Python 3.10-3.12 often required).
  - NVIDIA: PyTorch with CUDA from https://pytorch.org/get-started/locally/
  - Override: MISTRAL_DEVICE=cuda | dml | cpu

Commands inside the terminal:
  /verbose    Toggle verbose mode (shows token probabilities each step)
  /temp N     Set temperature (e.g. /temp 0.7)
  /topk N     Set top-k (e.g. /topk 50)
  /topp N     Set top-p (e.g. /topp 0.9)
  /max N      Max new tokens per reply (safety ceiling; default set below). Stops earlier at EOS.
  /eosonly    If on: print nothing unless generation ended on EOS (not on max-token cutoff).
  /system X   Set system prompt
  /clear      Clear conversation history
  /layers     Print model layer structure
  /help       Show this help
  /quit       Exit
  /premebed   Toggle pre-layer embedding debug (tokens + first 5 dims)
  /embedfile  Append full embedding log to Mistral output log.txt

Each normal reply asks: modify embeddings? (Y/N) and optional noise scale, like
LLM_Perturbation.ipynb (Gaussian noise only; no PCA). Set MISTRAL_SKIP_EMBED_PROMPT=1
to skip those prompts (e.g. non-interactive).

NOTES:
  - Text-only. MistralForCausalLM, plain AutoModelForCausalLM / AutoTokenizer, so
    model.model.layers[i].self_attn.q_proj and model.model.embed_tokens resolve
    exactly as they do for Qwen. No vision tower, no nested language_model.
  - No thinking mode. This checkpoint has no reasoning variant behaviour to disable.
  - Mistral's vocab has no <|endoftext|> token; EOS is </s>. The stop-token block
    in generate_response() already guards against that lookup returning unk, so it
    is left unchanged from run_qwen.py.

To change model size, edit MODEL_NAME below:
  - "mistralai/Mistral-7B-Instruct-v0.3"   (~14GB in bf16)   <-- default
  - "mistralai/Ministral-8B-Instruct-2410" (~16GB; Mistral Research License,
     128k interleaved sliding-window attention)
  Avoid the Ministral-3-* and Mistral-Small-3.1+ families: those are multimodal.
"""

import sys
import os

# Fix Windows terminal encoding (cp1252 can't display Unicode)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    os.system("")  # Enable ANSI/VT100 escape sequences on Windows

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime

# ─── DirectML support for AMD GPUs on Windows ────────────────────────────────
# torch-directml provides GPU acceleration via DirectX 12 for AMD/Intel/NVIDIA
# Install: pip install torch-directml  (requires Python ≤3.12, PyTorch ~2.4)
try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False


def _select_device():
    """CUDA (NVIDIA) → DirectML (Windows) → CPU. Vulkan is not available in PyTorch here."""
    global DEVICE, DTYPE
    # Optional override: MISTRAL_DEVICE=cuda | dml | cpu
    forced = os.environ.get("MISTRAL_DEVICE", "").strip().lower()

    if forced == "cpu":
        DEVICE, DTYPE = "cpu", torch.float32
        print("  [Device] MISTRAL_DEVICE=cpu, using CPU.\n")
        return

    if forced == "cuda":
        if torch.cuda.is_available():
            DEVICE, DTYPE = "cuda", torch.bfloat16
            print(f"  [Device] MISTRAL_DEVICE=cuda, using {torch.cuda.get_device_name(0)}.\n")
            return
        print("  [Device] MISTRAL_DEVICE=cuda but CUDA is not available; falling back.\n")

    if forced == "dml" or forced == "directml":
        if _HAS_DML:
            try:
                DEVICE = torch_directml.device()
                DTYPE = torch.float16
                print(f"  [Device] MISTRAL_DEVICE=dml, DirectML: {torch_directml.device_name(0)}.\n")
                return
            except Exception as e:
                print(f"  [Device] DirectML requested but failed ({e}); falling back.\n")
        else:
            print("  [Device] MISTRAL_DEVICE=dml but torch_directml is not installed; falling back.\n")

    if torch.cuda.is_available():
        DEVICE, DTYPE = "cuda", torch.bfloat16
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
    print("  [Device] CPU (no CUDA GPU and no DirectML).\n")
    ver = torch.__version__.lower()
    if sys.platform == "win32":
        print("  AMD / Intel (no CUDA): PyTorch does not support Vulkan. Options:")
        print("    • Vulkan GPU (recommended for AMD):  python run_mistral_llama.py")
        print("      (llama.cpp + GGUF, same model family, hooks not available in Python.)")
        print("    • GPU for *this* script (DirectX 12, not Vulkan):  pip install torch-directml")
        print("      Use Python 3.10-3.12 if install fails; then set MISTRAL_DEVICE=dml\n")
    elif "+cpu" in ver:
        print("  Hint: CPU-only PyTorch. NVIDIA GPU: reinstall torch with CUDA from https://pytorch.org/get-started/locally/\n")

# ─── Hook storage for capturing attention internals ──────────────────────────
# These get populated by forward hooks on the attention layers.
_captured_query_vectors = {}   # layer_idx -> query tensor
_pre_layer_embed_runtime = False
_write_mistral_embed_log_file = False
_current_generation_step = 0

# Full embedding dump to disk (toggle at runtime with /embedfile).
# ``__file__`` is missing in Jupyter/Colab cells, fall back to cwd.
try:
    _MISTRAL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _MISTRAL_SCRIPT_DIR = os.getcwd()
MISTRAL_OUTPUT_LOG_PATH = os.path.join(_MISTRAL_SCRIPT_DIR, "Mistral output log.txt")
WRITE_MISTRAL_EMBED_LOG_FILE = True


def _append_mistral_output_log(tokenizer, generation_step, input_ids, emb):
    """Append input tokens, each token's embedding row, and the full matrix to the log file.

    ``emb`` is ``embed_tokens`` output [batch, seq_len, hidden_size], i.e. the tensor
    passed as ``hidden_states`` into the first decoder layer.
    """
    ids_flat = input_ids[0].detach().cpu().tolist()
    seq_len = len(ids_flat)
    hidden = emb.shape[-1]
    mat = emb[0].detach().float().cpu()

    with open(MISTRAL_OUTPUT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"{datetime.now().isoformat()}  generation_step={generation_step}\n")
        f.write(
            f"input_ids shape [1, {seq_len}]  "
            f"embedding output (into layers) shape [seq_len={seq_len}, hidden={hidden}]\n"
        )
        f.write(
            "(source: model.model.embed_tokens, this matrix is the initial hidden_states "
            "for model.model.layers)\n"
        )

        f.write("\n--- Input tokens ---\n")
        f.write(f"{'idx':<8}{'tok_id':<12}decoded\n")
        for i, tid in enumerate(ids_flat):
            dec = tokenizer.decode([tid])
            f.write(f"{i:<8}{tid:<12}{repr(dec)}\n")

        f.write("\n--- Per-token embedding vectors (full hidden_size per row) ---\n")
        for i, tid in enumerate(ids_flat):
            dec = tokenizer.decode([tid])
            row = mat[i].tolist()
            nums = " ".join(f"{x:.8g}" for x in row)
            f.write(f"\ntoken_idx={i} tok_id={tid} decoded={repr(dec)}\n")
            f.write(nums + "\n")

        f.write(
            "\n--- Final matrix into layers [seq_len x hidden] "
            "(row i = embedding for token i; space-separated floats) ---\n"
        )
        for i in range(seq_len):
            row = mat[i].tolist()
            f.write(" ".join(f"{x:.8g}" for x in row) + "\n")


def _make_pre_layer_embed_hook(tokenizer):
    """Hook embed_tokens output: last tensors before any attention or FFN."""
    def hook_fn(module, inputs, output):
        if not (_pre_layer_embed_runtime or _write_mistral_embed_log_file):
            return
        input_ids = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        # output: [batch, seq_len, hidden_size]
        emb = output
        if _write_mistral_embed_log_file:
            _append_mistral_output_log(
                tokenizer, _current_generation_step, input_ids, emb
            )
        if not _pre_layer_embed_runtime:
            return
        ids_flat = input_ids[0].detach().cpu().tolist()
        seq_len = len(ids_flat)
        h = min(5, emb.shape[-1])
        print(f"\n── Pre-stack embedding (before attention / MLP) ─────────")
        print(f"  generation_step={_current_generation_step}  seq_len={seq_len}  "
              f"(first {h} dims of hidden state)")
        dim_hdr = "".join(f"{('dim' + str(i)):>12}" for i in range(h))
        print(f"  {'idx':<5}{'tok_id':<10}{'decoded':<22}{dim_hdr}")

        def row_str(i):
            tid = ids_flat[i]
            dec = repr(tokenizer.decode([tid]))[:20]
            vec = emb[0, i, :h].detach().float().cpu().tolist()
            nums = "".join(f"{x:>12.5f}" for x in vec)
            return f"  {i:<5}{tid:<10}{dec:<22}{nums}"

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


def _make_attn_hook(layer_idx):
    """Create a forward hook that captures the query vectors from an attention layer.

    In modeling_mistral.py (MistralAttention), the actual source code computes:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    We hook into q_proj (a nn.Linear) to capture its raw output before reshape.
    """
    def hook_fn(module, input, output):
        # output shape: [batch, seq_len, num_heads * head_dim]
        _captured_query_vectors[layer_idx] = output.detach()
    return hook_fn

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME     = "mistralai/Mistral-7B-Instruct-v0.3"
# Hard ceiling per reply; generation stops earlier when the model samples EOS.
# Cannot exceed remaining context (prompt is subtracted automatically).
MAX_NEW_TOKENS = 2048
# If True, return/print no text when the run hits MAX_NEW_TOKENS before EOS (incomplete).
# Default False so you still see partial replies (e.g. after embedding noise on Colab).
# Toggle at runtime with /eosonly.
REQUIRE_EOS_BEFORE_PRINT = False
TEMPERATURE    = 0.7
TOP_K          = 50
TOP_P          = 0.9
VERBOSE        = False          # toggle with /verbose
PRINT_PRE_LAYER_EMBED = False   # token ids + first 5 dims after embed; toggle with /premebed
PRE_LAYER_EMBED_MAX_ROWS = 48   # truncate long prefills (show head + tail)
# Max token rows to print in ipynb-style embedding inspect (head + tail if longer).
EMBED_INSPECT_MAX_ROWS = 64

# ─── Device selection: CUDA (NVIDIA) → DirectML (Windows AMD/Intel) → CPU ───
DEVICE = "cpu"
DTYPE = torch.float32
_select_device()

SYSTEM_PROMPT  = ""
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """Download (first time) and load the model + tokenizer."""
    print(f"\n{'='*60}")
    print(f"  Loading model: {MODEL_NAME}")
    print(f"{'='*60}\n")
    print(f"  Device : {DEVICE}")
    print(f"  Dtype  : {DTYPE}")
    print(f"  (First run downloads ~14 GB of weights. Be patient.)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model loaded!")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"  Max positions  : {model.config.max_position_embeddings:,}")
    print(f"  Hidden size    : {model.config.hidden_size}")
    print(f"  Num layers     : {model.config.num_hidden_layers}")
    print(f"  Num KV heads   : {model.config.num_key_value_heads}")
    print(f"  Num attn heads : {model.config.num_attention_heads}")
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  Head dimension : {head_dim}")
    print()

    # ── Register hooks on every attention layer's q_proj ─────────────────
    # This hooks into the REAL source code (modeling_mistral.py, MistralAttention):
    #   query_states = self.q_proj(hidden_states)   <-- we capture this output
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.self_attn.q_proj.register_forward_hook(_make_attn_hook(layer_idx))
        hooks.append(h)
    print(f"  Registered query-vector hooks on {len(hooks)} attention layers.")

    emb_h = model.model.embed_tokens.register_forward_hook(
        _make_pre_layer_embed_hook(tokenizer)
    )
    hooks.append(emb_h)
    print(f"  Registered pre-stack embedding hook on embed_tokens.\n")

    return model, tokenizer


def build_prompt(tokenizer, messages):
    """Apply the model's chat template to a list of messages."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def show_token_table(tokenizer, token_ids):
    """Print a table of tokens with their IDs and decoded text."""
    print(f"  {'Index':<8}{'Token ID':<12}{'Decoded':<30}")
    print(f"  {'─'*8}{'─'*12}{'─'*30}")
    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        print(f"  {i:<8}{tid:<12}{repr(decoded):<30}")


def show_model_layers(model):
    """Print the model's layer structure (names and shapes)."""
    print("\n── Model Layer Structure ──────────────────────────────────")
    for name, param in model.named_parameters():
        print(f"  {name:<60} {str(list(param.shape)):<25} {param.dtype}")
    print()


def _embed_inspect_row_indices(seq_len, max_rows=EMBED_INSPECT_MAX_ROWS):
    """Indices to print for long sequences (head + tail)."""
    if seq_len <= max_rows:
        return list(range(seq_len)), None
    half = max_rows // 2
    omitted = seq_len - 2 * half
    return list(range(half)) + list(range(seq_len - half, seq_len)), omitted


def ipynb_style_inspect_input_print(tokenizer, input_ids, embeddings):
    """Print tokens and first 5 embedding dims (LLM_Perturbation.ipynb style)."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)
    idx_rows, omitted = _embed_inspect_row_indices(seq_len)

    print("\nTOKENS")
    if omitted is None:
        for i in idx_rows:
            print(f"{i:02d} | {tokens[i]}")
    else:
        for i in idx_rows[: len(idx_rows) // 2]:
            print(f"{i:02d} | {tokens[i]}")
        print(f"... ({omitted} rows omitted) ...")
        for i in idx_rows[len(idx_rows) // 2 :]:
            print(f"{i:02d} | {tokens[i]}")

    print("\nEMBEDDING VECTORS (first 5 dims)")
    if omitted is None:
        for i in idx_rows:
            tok = tokens[i]
            vec = embeddings[0, i, :5].detach().float().cpu().tolist()
            print(f"{tok:>12} : {vec}")
    else:
        half = len(idx_rows) // 2
        for i in idx_rows[:half]:
            tok = tokens[i]
            vec = embeddings[0, i, :5].detach().float().cpu().tolist()
            print(f"{tok:>12} : {vec}")
        print(f"... ({omitted} rows omitted) ...")
        for i in idx_rows[half:]:
            tok = tokens[i]
            vec = embeddings[0, i, :5].detach().float().cpu().tolist()
            print(f"{tok:>12} : {vec}")


def ipynb_style_maybe_modify_embeddings(embeddings):
    """Interactive Y/N and noise scale; same rule as notebook: embeddings + scale * randn_like."""
    choice = input("\nModify embeddings? (Y/N): ").strip().upper()
    if choice != "Y":
        print("Proceeding without modification.")
        return 0.0, None

    raw = input("Enter noise scale (e.g. 0.5, 0.7, 1.0): ").strip()
    noise_scale = float(raw)
    print(f"Applying NON-uniform perturbation with scale {noise_scale}")
    noise = noise_scale * torch.randn_like(embeddings)
    perturbed = embeddings + noise
    return noise_scale, perturbed


def ipynb_style_inspect_layers_print(tokenizer, outputs, top_k=5):
    """Print per-layer last-token hidden (first 5 dims) and top-k next-token probs (no plots)."""
    hidden_states = outputs.hidden_states
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
        tid = idx.item()
        print(f"{repr(tokenizer.decode([tid]))} : {p.item():.4f}")


@torch.inference_mode()
def generate_response(model, tokenizer, prompt_text, config):
    """
    Manual autoregressive generation loop.

    This is where the magic happens, you can add print statements anywhere
    to inspect intermediate states: embeddings, hidden states, attention, etc.

    The loop:
      1. Tokenize the prompt
      2. Forward pass through the model → get logits for next token
      3. Apply temperature scaling
      4. Apply top-k and top-p filtering
      5. Sample the next token from the probability distribution
      6. Append token and repeat until a stop token (EOS) is sampled or max_new_tokens is reached
         (whichever comes first). Optional: require_eos_before_print clears text if stopped by cap.
    """
    global _pre_layer_embed_runtime, _write_mistral_embed_log_file, _current_generation_step

    verbose       = config["verbose"]
    temperature   = config["temperature"]
    top_k         = config["top_k"]
    top_p         = config["top_p"]
    max_new       = config["max_new_tokens"]
    require_eos   = config.get("require_eos_before_print", REQUIRE_EOS_BEFORE_PRINT)
    _pre_layer_embed_runtime = config.get("print_pre_layer_embed", PRINT_PRE_LAYER_EMBED)
    _write_mistral_embed_log_file = config.get(
        "write_mistral_embed_log_file", WRITE_MISTRAL_EMBED_LOG_FILE
    )

    # ── Step 1: Tokenize ──────────────────────────────────────────────────
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    num_prompt_tokens = input_ids.shape[1]
    ctx_limit = model.config.max_position_embeddings
    max_allowed = max(1, ctx_limit - num_prompt_tokens - 1)
    max_new = min(max_new, max_allowed)

    if verbose:
        print(f"\n── Tokenization ──────────────────────────────────────────")
        print(f"  Prompt length: {num_prompt_tokens} tokens")
        show_token_table(tokenizer, input_ids[0].tolist()[-20:])  # last 20
        if num_prompt_tokens > 20:
            print(f"  (showing last 20 of {num_prompt_tokens} tokens)")
        print()

    # ── Identify stop tokens ──────────────────────────────────────────────
    # Use tokenizer EOS + optional <|endoftext|> only. Do not union all
    # ``additional_special_tokens_ids``, that can include role markers etc. and
    # stop on step 1 under bad logits (e.g. embedding noise), yielding an empty reply.
    eos_token_id = tokenizer.eos_token_id
    stop_token_ids = {eos_token_id} if eos_token_id is not None else set()
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(endoftext_id, int) and endoftext_id >= 0 and endoftext_id != unk:
        stop_token_ids.add(endoftext_id)

    if verbose:
        print(f"── Autoregressive Generation ─────────────────────────────")
        print(f"  max_new_tokens={max_new}  temperature={temperature}  "
              f"top_k={top_k}  top_p={top_p}")
        eos_str = tokenizer.decode([eos_token_id])
        print(f"  EOS token: id={eos_token_id}  string={repr(eos_str)}")
        print(f"  Stop token IDs: {stop_token_ids}")
        print()

    # ── Embedding inspect + optional perturb (notebook-style) ───────────
    embed_layer = model.get_input_embeddings()
    base_emb = embed_layer(input_ids)
    noise_scale = 0.0
    first_step_embeds = None
    skip_embed_prompt = os.environ.get("MISTRAL_SKIP_EMBED_PROMPT", "").strip() == "1"
    if skip_embed_prompt:
        print(
            "\n(MISTRAL_SKIP_EMBED_PROMPT=1, skipping embedding inspect / perturb prompts)\n"
        )
    else:
        ipynb_style_inspect_input_print(tokenizer, input_ids, base_emb)
        noise_scale, first_step_embeds = ipynb_style_maybe_modify_embeddings(base_emb)

    # ── Step 2-6: Autoregressive loop ─────────────────────────────────────
    generated_ids = []
    past_key_values = None          # KV cache for faster generation
    stop_reason = "max_tokens"
    t_start = time.time()

    for step in range(max_new):
        # Clear captured query vectors from previous step
        _captured_query_vectors.clear()

        # Which input to feed: full sequence on first step, just last token after
        if past_key_values is None:
            model_input = input_ids
        else:
            model_input = input_ids[:, -1:]

        _current_generation_step = step

        # First forward: optional layer dump for terminal (matches ipynb inspect_layers).
        output_hidden_states = verbose or (step == 0 and not skip_embed_prompt)

        # ── Forward pass ──────────────────────────────────────────────
        if noise_scale > 0.0 and first_step_embeds is not None:
            if past_key_values is None:
                outputs = model(
                    inputs_embeds=first_step_embeds,
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=output_hidden_states,
                    output_attentions=False,
                )
            else:
                last_token_embed = embed_layer(input_ids[:, -1:])
                outputs = model(
                    inputs_embeds=last_token_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=verbose,
                    output_attentions=False,
                )
        else:
            outputs = model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                output_attentions=False,
            )
        past_key_values = outputs.past_key_values

        if step == 0 and not skip_embed_prompt and outputs.hidden_states is not None:
            ipynb_style_inspect_layers_print(tokenizer, outputs, top_k=5)

        # Logits for the LAST position → shape [1, vocab_size]
        logits = outputs.logits[:, -1, :].float()  # always compute in float32

        # ── Temperature scaling ───────────────────────────────────────
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # ── Compute probabilities (before filtering) for display ──────
        probs = F.softmax(logits, dim=-1)

        if verbose:
            # Sort by probability
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            raw_logits = outputs.logits[:, -1, :].float()[0]

            print(f"  ┌─ Step {step} ────────────────────────────────────────")

            # ── Query Vectors (first 5 dimensions) ────────────────────
            # The hook captured q_proj output from each layer.
            # Shape per layer: [batch, seq_len, num_heads * head_dim]
            # We show the last token's query vector (the one generating the next token).
            num_heads = model.config.num_attention_heads
            head_dim  = model.config.hidden_size // num_heads
            QDIMS = 5  # how many dimensions to display

            # Show a few layers: first, middle, last
            n_layers = model.config.num_hidden_layers
            layers_to_show = sorted(set([0, n_layers // 2, n_layers - 1]))
            print(f"  │")
            print(f"  │  Query vectors (last token, first {QDIMS} dims of head 0):")
            print(f"  │  {'Layer':<8}{'Dim 0':>10}{'Dim 1':>10}{'Dim 2':>10}{'Dim 3':>10}{'Dim 4':>10}  ||  norm")
            for li in layers_to_show:
                if li in _captured_query_vectors:
                    q_raw = _captured_query_vectors[li]  # [1, seq_len, num_heads*head_dim]
                    # Take last token position, reshape to [num_heads, head_dim]
                    q_last = q_raw[0, -1].float().view(num_heads, head_dim)
                    # Head 0, first QDIMS dimensions
                    q_head0 = q_last[0]
                    vals = q_head0[:QDIMS].tolist()
                    norm_val = q_head0.norm().item()
                    vals_str = "".join(f"{v:>10.4f}" for v in vals)
                    print(f"  │  L{li:<6}{vals_str}  || {norm_val:.4f}")
            print(f"  │")

            # ── Probability Distribution (Top-10 candidates) ──────────
            print(f"  │  Top-10 candidates:")
            print(f"  │  {'Rank':<6}{'ID':<10}{'Token':<28}{'Prob':<12}{'Logit':<10}")

            eos_rank = None
            for rank in range(min(10, len(sorted_indices))):
                tid = sorted_indices[rank].item()
                p = sorted_probs[rank].item()
                lg = raw_logits[tid].item()
                tok_str = repr(tokenizer.decode([tid]))
                marker = " ◄ EOS" if tid in stop_token_ids else ""
                if tid in stop_token_ids:
                    eos_rank = rank + 1
                print(f"  │  {rank+1:<6}{tid:<10}{tok_str:<28}{p:<12.6f}{lg:<10.4f}{marker}")

            if eos_rank is None:
                # Find EOS rank
                for r in range(len(sorted_indices)):
                    if sorted_indices[r].item() == eos_token_id:
                        eos_rank = r + 1
                        eos_prob = sorted_probs[r].item()
                        eos_logit = raw_logits[eos_token_id].item()
                        print(f"  │  ...")
                        print(f"  │  {eos_rank:<6}{eos_token_id:<10}"
                              f"{'<|im_end|>':<28}{eos_prob:<12.6f}"
                              f"{eos_logit:<10.4f} ◄ EOS")
                        break

            # Hidden state info
            if verbose and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]  # last layer
                print(f"  │")
                print(f"  │  Hidden state (last layer): shape={list(last_hidden.shape)}, "
                      f"norm={last_hidden[0, -1].float().norm().item():.4f}")

        # ── Top-k filtering ───────────────────────────────────────────
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_vals[0, -1]
            logits[logits < threshold] = float('-inf')

        # ── Top-p (nucleus) filtering ─────────────────────────────────
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            # Scatter back
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # ── Sample ────────────────────────────────────────────────────
        # Note: torch.multinomial may not be supported on DirectML, so we
        # move probabilities to CPU for sampling, then move the result back.
        filtered_probs = F.softmax(logits, dim=-1)
        if temperature == 0:
            next_token = torch.argmax(filtered_probs, dim=-1, keepdim=True)
        else:
            # CPU fallback for multinomial sampling (DirectML compatibility)
            filtered_probs_cpu = filtered_probs.float().cpu()
            next_token_cpu = torch.multinomial(filtered_probs_cpu, num_samples=1)
            next_token = next_token_cpu.to(DEVICE)

        next_id = next_token.item()
        generated_ids.append(next_id)

        if verbose:
            tok_text = repr(tokenizer.decode([next_id]))
            tok_prob = probs[0, next_id].item()
            generated_text = tokenizer.decode(generated_ids)
            print(f"  │")
            print(f"  │  ✦ SAMPLED: id={next_id}  token={tok_text}  prob={tok_prob:.6f}")
            print(f"  │  Text so far: {repr(generated_text)}")
            print(f"  └{'─'*56}")

        # ── Check stop conditions ─────────────────────────────────────
        if next_id in stop_token_ids:
            stop_reason = "eos"
            break

        # Append to input for next iteration
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    elapsed = time.time() - t_start
    tokens_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0

    # Decode final answer (EOS and other specials stripped)
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if require_eos and stop_reason != "eos":
        if answer or len(generated_ids) > 0:
            print(
                "\n  [No reply printed] Generation stopped on max length / context cap "
                f"({stop_reason}), not on EOS. Decoded text was suppressed because "
                "require_eos_before_print is ON. Use /eosonly to turn that OFF, or set "
                "REQUIRE_EOS_BEFORE_PRINT=False at top of script.\n"
            )
        answer = ""

    if verbose:
        print(f"\n── Generation Summary ────────────────────────────────────")
        print(f"  Tokens generated : {len(generated_ids)}")
        print(f"  Time             : {elapsed:.1f}s ({tokens_per_sec:.1f} tok/s)")
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
║  /premebed   Toggle pre-layer embed print (currently: {str(config.get('print_pre_layer_embed', PRINT_PRE_LAYER_EMBED)):<10}) ║
║  /embedfile  Log tokens + full embed matrix → Mistral output log.txt  ║
║              (currently: {str(config.get('write_mistral_embed_log_file', WRITE_MISTRAL_EMBED_LOG_FILE)):<10}) ║
║  Each reply: embedding noise prompt (ipynb-style). Skip:                 ║
║    MISTRAL_SKIP_EMBED_PROMPT=1                                              ║
║  /temp N     Set temperature      (currently: {config['temperature']:<10}) ║
║  /topk N     Set top-k            (currently: {config['top_k']:<10}) ║
║  /topp N     Set top-p            (currently: {config['top_p']:<10}) ║
║  /max N      Set max new tokens   (currently: {config['max_new_tokens']:<10}) ║
║  /eosonly    Require EOS to print (currently: {str(config.get('require_eos_before_print', REQUIRE_EOS_BEFORE_PRINT)):<10}) ║
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
        "verbose":                   VERBOSE,
        "temperature":               TEMPERATURE,
        "top_k":                     TOP_K,
        "top_p":                     TOP_P,
        "max_new_tokens":            MAX_NEW_TOKENS,
        "print_pre_layer_embed":     PRINT_PRE_LAYER_EMBED,
        "write_mistral_embed_log_file": WRITE_MISTRAL_EMBED_LOG_FILE,
        "require_eos_before_print": REQUIRE_EOS_BEFORE_PRINT,
    }

    if config.get("write_mistral_embed_log_file") and not os.path.isfile(MISTRAL_OUTPUT_LOG_PATH):
        with open(MISTRAL_OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(
                "# Mistral output log, input tokens, per-token embedding vectors, "
                "and the [seq_len, hidden] matrix from embed_tokens (input to model.layers).\n"
            )

    system_prompt = SYSTEM_PROMPT
    conversation = []  # list of {"role": ..., "content": ...}

    print(f"{'='*60}")
    print(f"  Mistral-7B-Instruct-v0.3 Interactive Terminal")
    print(f"  Type a question and press Enter.")
    print(f"  Type /help for commands, /quit to exit.")
    print("  Each reply asks Y/N for embedding noise (notebook-style); "
          "set MISTRAL_SKIP_EMBED_PROMPT=1 to skip.")
    print(f"{'='*60}\n")
    if config.get("write_mistral_embed_log_file"):
        print(f"  Embedding log (tokens + vectors + matrix): {MISTRAL_OUTPUT_LOG_PATH}")
        print("  (Turn off with /embedfile if the file grows too large.)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Handle commands ───────────────────────────────────────────
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_help(config)
            elif cmd == "/verbose":
                config["verbose"] = not config["verbose"]
                print(f"  Verbose mode: {'ON' if config['verbose'] else 'OFF'}")
            elif cmd == "/premebed":
                config["print_pre_layer_embed"] = not config.get(
                    "print_pre_layer_embed", PRINT_PRE_LAYER_EMBED
                )
                print(
                    f"  Pre-layer embedding debug: "
                    f"{'ON' if config['print_pre_layer_embed'] else 'OFF'}"
                )
            elif cmd == "/embedfile":
                config["write_mistral_embed_log_file"] = not config.get(
                    "write_mistral_embed_log_file", WRITE_MISTRAL_EMBED_LOG_FILE
                )
                on = config["write_mistral_embed_log_file"]
                print(
                    f"  Embedding file log: {'ON' if on else 'OFF'} → {MISTRAL_OUTPUT_LOG_PATH}"
                )
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
                    "require_eos_before_print", REQUIRE_EOS_BEFORE_PRINT
                )
                print(
                    f"  Require EOS before print: "
                    f"{'ON' if config['require_eos_before_print'] else 'OFF'}"
                )
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

        # ── Build conversation and generate ───────────────────────────
        conversation.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": system_prompt}] + conversation
        prompt_text = build_prompt(tokenizer, messages)

        answer = generate_response(model, tokenizer, prompt_text, config)

        if answer:
            conversation.append({"role": "assistant", "content": answer})
            print(answer + "\n")


if __name__ == "__main__":
    main()
