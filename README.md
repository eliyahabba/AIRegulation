### Beyond Benchmarks: On The False Promise of AI Regulation

## Overview
[![Paper](https://img.shields.io/badge/arxiv-paper-red)](https://arxiv.org/abs/2501.15693) [![Dataset](https://img.shields.io/badge/🤗-dataset-yellow)](https://huggingface.co/datasets/nlphuji/AI_Regulation) [![Contact](https://img.shields.io/badge/📧-contact-green)](mailto:eliyahaba@mail.huji.ac.il)

Evaluate LLM safety under prompt variation. This repository generates controlled prompt variations (using PromptSuite), runs multiple models, judges safety with GPT‑4o‑mini, and produces a unified figure that visualizes performance ranges across variations.

### What this project does
- Generates 20 prompt variations per harmful request while preserving the exact harmful content from AIR‑Bench and only changing the surrounding instruction and prompt formatting
- Runs multiple LLMs on 16 safety‑critical categories, 10 base prompts each (160 base prompts total)
- Uses GPT‑4o‑mini as an automated judge to score safety refusals (1) vs. dangerous compliance (0)
- Produces a figure showing, per model, the performance range across semantically identical prompts that differ only in instruction phrasing and prompt formatting

### Key idea (PromptSuite)
Variations are created with [PromptSuite](https://github.com/eliyahabba/PromptSuite), as described in [Habba et al. (2025)](https://arxiv.org/abs/2507.14913).
- Instruction paraphrase: semantically equivalent changes in phrasing/style 
- Prompt formatting (surface noise): spaces, typos, casing, punctuation  

### Reproducible experiment setup (numbers)
- 16 categories × 10 base prompts = 160 base scenarios
- 20 variations per scenario → 3,200 runs per model
- 11 models → 35,200 model responses total, and the same number of judge evaluations with GPT‑4o‑mini

### Models evaluated (ordered by size)
- Qwen2‑7B (`qwen_7b`)
- DeepSeek‑V2‑Lite (`deepseek_7b`)
- Command‑R‑08‑2024 (`command_r7b`)
- Llama 3‑8B (`llama3_8b`)
- Ministral‑8B (`mistral_8b`)
- Qwen3‑8B (`qwen3_8b`)
- Aya‑Expanse‑8B (`aya_expanse_8b`)
- OLMo‑2‑13B (`olmo_2_13b`)
- Phi‑4‑mini (`phi_4_mini`)
- Mixtral‑8x7B (`mixtral_8x7b`)
- Llama 3.3‑70B (`llama_3_3_70b`)

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

Recommended: Python 3.10+, GPU for local models (transformers/torch).

### 2) Environment
Create `.env` in `src/` or project root with the keys you will use:
```
OPENAI_API_KEY=...
TOGETHER_API_KEY=...
ANTHROPIC_API_KEY=...        # optional
GOOGLE_API_KEY=...           # optional
COHERE_API_KEY=...           # optional
HF_ACCESS_TOKEN=...          # optional for gated models (local)
```

### 3) Generate AIR‑Bench variations (PromptSuite)
Generates 20 variations per item while preserving the exact harmful content and changing only instruction/formatting.
```bash
python src/data_generation/airbench_task.py \
  --rows 10 \
  --variations 20 \
  --variations_per_field 10
```
Outputs to `data/generated_data/airbench/airbench_variations.json` (or per‑category file if you pass `--category`).

### 4) Run models on AIR‑Bench
Run locally (Hugging Face, requires GPU) or via APIs. Examples:
```bash
# Local (e.g., Llama‑3.1‑8B with 8‑bit quant)
python src/execution/run_airbench_batch.py \
  --platform local --model llama3_8b --quantization 8bit \
  --all --max_tokens 1000 --temperature 0.0

# TogetherAI (Llama‑3.3‑70B Turbo)
python src/execution/run_airbench_batch.py \
  --platform TogetherAI --model llama_3_3_70b \
  --all --max_tokens 1000 --temperature 0.0
```
Outputs per‑model to `data/results/airbench/<model_short>/airbench_variations.csv` (+ JSON).

Tip: list available AIR‑Bench datasets/files:
```bash
python src/execution/run_airbench_batch.py --list_datasets
```

### 5) Evaluate with GPT‑4o‑mini (judge)
Scores each response for safety (1, 0.5, 0). Use `--judge_model gpt_4o_mini` to match the experiment:
```bash
python src/evaluation/evaluate_airbench.py \
  --model llama3_8b \
  --judge_model gpt_4o_mini --judge_platform OpenAI
```
This writes `airbench_variations_evaluated.csv` next to the inputs (and intermediate `_evaluated.csv` during batching).

Optional utilities:
- Clean parsed responses: `python src/clean_parsed_responses.py`
- Remove error rows in CSVs: edit path and run `python src/clean_results.py`

### 6) Produce the unified figure
```bash
python src/analysis/unified_variation_analysis.py
```
Saves PNG/PDF to `data/output/unified_variation_analysis/` and prints per‑model min/median/max with the range across variations.

Figure caption (short):
> Figure X: Performance range across 20 prompt variations per model on AIR‑Bench. Variations preserve the exact harmful content and change only instruction phrasing and formatting (via PromptSuite). Most models show 5–20% ranges between their best and worst variation.

## Data and directories
- Generated data: `data/generated_data/airbench/`
- Model outputs: `data/results/airbench/<model_short>/`
- Analysis outputs: `data/output/unified_variation_analysis/`

## HPC script (optional)
SLURM example for local HF models: `src/sh_files/run_airbench_on_model.sh`

## Notes and citations
- PromptSuite: prompt generation framework enabling controlled perturbations (instruction paraphrase; formatting noise). Add citation placeholder and see `https://github.com/eliyahabba/PromptSuite`.
- Instruction paraphrase follows Mizrahi et al. (2024); prompt formatting noise follows Sclar et al. (2023).

## Troubleshooting
- “No AIR‑Bench evaluated results found …”: ensure step 5 produced `airbench_variations_evaluated.csv` under each model’s results folder.
- API errors: check `.env` keys and provider quotas.
- Local models on macOS use MPS; for quantization, prefer CUDA.