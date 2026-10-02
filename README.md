# Jailbreaking Open-Weight LLMs via Random Embedding Perturbations: runs, labels and code

Code and data for the paper
*Jailbreaking Open-Weight LLMs via Random Embedding Perturbations*.

The repository holds every model response generated for the paper, the safe / unsafe /
degenerate label of each response, the Colab notebooks that produced them, and the inference
scripts the notebooks import. This covers the Perturbed Embedding Vector (PEV) attack and the
five other jailbreak methods compared in the paper. All jailbreak rates and wall-clock times in
the paper are computed from the files in `results/`.

## Layout

```
code/
  inference/             run_<model>.py: model loading, prompt construction and the manual
                         generation loop that every PEV notebook imports
  ours/<model>/          PEV notebooks for one model
  ours/<model>/fixed_peak_sigma/
                         notebooks for the fixed peak sigma runs (GLM-4-9B, Phi-4-mini,
                         Qwen2.5-1.5B); their logs are in results/<model>/ours/fixed_peak_sigma/
  igcg/                  I-GCG, one notebook per model
  latentfusion/          LatentFusion, one notebook per model
  refusalcones/          RefusalCones, one notebook per model
  softprompt/            SoftPrompt notebooks (Llama-3.1-8B and Mistral-7B)
  figures/               notebook that draws the paper figures
results/<model>/
  baseline/              direct responses to the unperturbed prompts (.txt log) and their
                         labels (.xlsx / .csv); these give the baseline jailbreak rate
  ours/raw_responses/    PEV response logs from the sigma sweeps, split by prompt range
  ours/classified/       labels for those responses, split the same way
  ours/fixed_peak_sigma/ PEV runs at the fixed peak sigma of the model (GLM-4-9B, Phi-4-mini,
                         Qwen2.5-1.5B): the full 100-prompt sweep and the SELECT re-runs
  igcg/ latentfusion/ neurostrike/ refusalcones/ softprompt/
                         responses (.txt) and labels (.xlsx) for the other methods
```

Models: `Qwen2.5-1.5B`, `SmolLM3-3B`, `Phi-4-mini`, `Mistral-7B-Instruct`, `Llama-3.1-8B`,
`GLM-4-9B`, the instruction-tuned chat variants named in Sec. 4 of the paper.

Where each method's code is:

| Method       | Location |
|--------------|----------|
| PEV          | `code/ours/<model>/`; fixed peak sigma runs in `code/ours/<model>/fixed_peak_sigma/`; shared inference code in `code/inference/` |
| I-GCG        | `code/igcg/`, one notebook per model |
| LatentFusion | `code/latentfusion/`, one notebook per model |
| RefusalCones | `code/refusalcones/`, one notebook per model |
| SoftPrompt   | `code/softprompt/`, Llama-3.1-8B and Mistral-7B |
| NeuroStrike  | inside the PEV notebooks `code/ours/Llama-3.1-8B/LLama31_perturbation_p76_to_p100.ipynb` (setup cell also in `LLama31_perturbation_p1_to_p25.ipynb`) and `code/ours/SmolLM3-3B/run_smollm3_batched_p26_p50.ipynb`; responses go to `results/<model>/neurostrike/` |

## Inference scripts

`code/inference/run_<model>.py` is one file per model (`run_qwen.py`, `run_smollm3.py`,
`run_phi4mini.py`, `run_mistral.py`, `run_llama.py`, `run_glm.py`). Each loads the model with
HuggingFace Transformers, applies the chat template with an empty system message, exposes
`load_model()`, `build_prompt()` and `encode_prompt()`, and runs a manual autoregressive
generation loop with a hook on the input embeddings. The notebooks import the script for their
model and inject the Gaussian noise through that hook. Run alone, a script opens an
interactive terminal for a single model; the `/verbose` and `/embedfile` commands print token
tables and embedding vectors. Device and behaviour switches are environment variables
documented in the header of each file. The Llama script needs a Hugging Face token because the
checkpoint is gated; the token is read from the environment and never stored.

## File conventions

- Response logs (`.txt`) are append-only logs written by the notebooks. Each entry records
  the prompt index, the noise level sigma where applicable, the run number, and the full
  model response. File names carry the prompt range (`p001_to_p025` means prompts 1 to 25 of
  the 100 JailbreakBench harmful prompts) and the UTC timestamp of the run. Wall-clock time is
  written after every batch of 20 runs per prompt and at the end of the experiment.
- Label files (`.xlsx`, `.csv`) hold one row per response with its label, safe, unsafe or
  degenerate, as defined in Sec. 4 of the paper. A prompt counts as jailbroken if any of its
  runs is labelled unsafe. Files named `*_categorization*` or `*_p01_25`-style are the
  labelled versions of the log with the same prompt range.
- Fixed peak sigma runs (`ours/fixed_peak_sigma/`) are named
  `JBB_<model>_sigma0pXXX_..._<timestamp>.txt`; `sigma0p003` means sigma = 0.003. The
  `p001_to_p100` file is the full 100-prompt, 20-run sweep at that sigma. The
  `SELECT_..._nNN` files are follow-up runs at the same sigma on the prompts listed in the
  file name, with NN runs per prompt.
- Files containing `HIGHSIGMA` are the runs at larger sigma discussed in Sec. 4.1.
- Files containing `SELECT` are additional runs on the prompts listed in the file name, for
  prompts that needed more than 20 runs to jailbreak. The header line of every log records the
  sigma values, the number of runs, the prompt ids and the random seed.
- `Llama-3.1-8B/baseline/fixed_baseline.xlsx` and the `fixed_*` label files are the label files
  after manual verification and supersede earlier versions.

## Response classification

Responses were classified by Claude Opus 4.6 with the JailbreakBench judge instructions, given
the log file with the prompt, run number and response. Every response labelled unsafe was then
checked by hand. The label files in `results/` are the verified labels.

## Reproducing a run

Each notebook under `code/ours/<model>/` was run on a single NVIDIA A100-SXM4-80GB in Google
Colab. It imports the matching `run_<model>.py`, loads the JailbreakBench harmful prompts,
maps each prompt through the model's input embedding layer, adds independent Gaussian noise
N(0, sigma^2) to the full prompt embedding, and passes the perturbed embedding to the
transformer layers. For each prompt it samples 20 noise matrices in one batch and appends every
response to the log. Decoding parameters and the per-model sigma are listed in Sec. 4 and
Tab. 3 of the paper and set at the top of each notebook. The Hugging Face token is requested
interactively and is not stored in this repository. The notebooks for the other methods follow
the same pattern with the corresponding attack substituted.

The jailbreak rates in the paper can be recomputed from the label files alone, without running
any model.

## Citation

If you use this code or data, please cite the paper:

```
@article{dubey2026pev,
  title   = {Jailbreaking Open-Weight LLMs via Random Embedding Perturbations},
  author  = {Dubey, Abhinav},
  journal = {arXiv preprint},
  year    = {2026}
}
```
