# Invoice/Document Extraction — LoRA Fine-Tune

Fine-tunes a small open-source instruction model with LoRA to extract structured JSON
fields from invoice text, and reports a real, measured before/after accuracy comparison
against the base model's zero-shot performance.

Built with Claude Code as a coding assistant.

## Design decisions

1. **Base model: `Qwen/Qwen2.5-1.5B-Instruct`.** Training a 7-8B model needs more VRAM than my
   4 GB GTX 1650 has, even with 4-bit QLoRA, and I had no AWS budget. I used a ~1.5B-parameter
   ungated model so the full pipeline could run locally with real measured results. Qwen2.5-1.5B-Instruct
   uses the same attention module names (`q_proj/k_proj/v_proj/o_proj`) so the LoRA config works unchanged.
2. **Synthetic data generator: `openai/gpt-oss-20b` via the Groq API.** I didn't have an OpenAI or
   Anthropic API key. Went through
   two intermediate providers first (a Gemini key, capped at 20 requests/day on the fast
   models; then a slower Gemma model on the same key, ~30-50s/call) before I switched to Groq, which responds in well under a second. See
   [`data/synthetic/GENERATION_LOG.md`](data/synthetic/GENERATION_LOG.md) for the full
   story, exact prompt, and model.
3. **No AWS SageMaker training job / `infra/sagemaker_train.py` not built.** No AWS
   credentials or budget were authorized. Training ran locally instead (`src/train/train_lora.py`), which is why the smaller base model
   was necessary.
4. **No `HF_TOKEN` needed.** Qwen2.5-1.5B-Instruct is not gated on Hugging Face Hub.
5. **Synthetic dataset size: 207 instead of 1000.** Even on Groq, the free-tier
   8000-tokens/minute cap throttled sustained throughput. I stopped at 207 examples rather than waiting for the full 1000. This
   produced 21 held-out synthetic test examples instead of 100 (train=617, val=69 — see
   [`data/synthetic/GENERATION_LOG.md`](data/synthetic/GENERATION_LOG.md) for exact
   counts). Expect noisier metrics, especially line-item precision/recall, than the full
   design would give.

## Data

- SROIE 2019 source: Hugging Face Datasets [`darentang/sroie`](https://huggingface.co/datasets/darentang/sroie)
  (ICDAR2019 SROIE, Task 3 — Key Information Extraction). SROIE ground truth only covers
  `company`/`date`/`address`/`total`; this repo uses `company`→`vendor`, `date`→
  `invoice_date`, `total`→`total` only. Line items and invoice numbers are **not** present
  in SROIE ground truth — they come entirely from the synthetic set. `address` is dropped
  (not part of the target schema).
- SROIE license: confirmed via the HF Hub API that `darentang/sroie` carries no license
  tag at all — redistribution terms are unclear, so
  `data/raw/sroie/` is gitignored, and so are `data/processed/train.jsonl`,
  `val.jsonl`, and `test_sroie.jsonl` (they embed real SROIE OCR text, since train/val mix
  synthetic + SROIE examples together and can't be cleanly split apart after the fact).
  Only `data/synthetic/synthetic_raw.jsonl` (pure synthetic) and
  `data/processed/test_synthetic.jsonl` (pure synthetic) are committed. Anyone reproducing
  the SROIE-eval numbers must download SROIE themselves and rerun
  `python -m src.data.prepare_dataset --skip-generation` — deterministic (seed=42) given
  the committed `synthetic_raw.jsonl`.
- Synthetic invoices: see
  [`data/synthetic/GENERATION_LOG.md`](data/synthetic/GENERATION_LOG.md).

## Repository structure

Top-level folders: `data/`, `models/`, `results/`, `src/`, `tests/`.

## Setup

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (see .env.example)
```

## Reproduce

```bash
# 1. Data pipeline — downloads SROIE, generates synthetic data, writes processed JSONL
.venv/Scripts/python.exe -m src.data.prepare_dataset

# 2. Baseline (zero-shot) eval — MUST run before any fine-tuning
.venv/Scripts/python.exe -m src.eval.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --mode zero-shot \
    --test data/processed/test_synthetic.jsonl --out results/baseline_synthetic.json
.venv/Scripts/python.exe -m src.eval.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --mode zero-shot \
    --test data/processed/test_sroie.jsonl --out results/baseline_sroie.json

# 3. Fine-tune
.venv/Scripts/python.exe -m src.train.train_lora --train data/processed/train.jsonl \
    --val data/processed/val.jsonl --output-dir models/invoice-lora-v1

# 4. Fine-tuned eval
.venv/Scripts/python.exe -m src.eval.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --mode finetuned \
    --adapter models/invoice-lora-v1 --test data/processed/test_synthetic.jsonl \
    --out results/finetuned_synthetic.json
.venv/Scripts/python.exe -m src.eval.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --mode finetuned \
    --adapter models/invoice-lora-v1 --test data/processed/test_sroie.jsonl \
    --out results/finetuned_sroie.json

# 5. Demo (CLI)
.venv/Scripts/python.exe -m src.demo.cli --input path/to/invoice.txt

# 5b. Demo (Gradio web UI, optional — textbox in, JSON out)
.venv/Scripts/python.exe -m src.demo.app
```

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -v --basetemp=.pytest_tmp
```

(`--basetemp` works around a Windows AppData permissions issue with pytest's default temp
dir on this machine; not needed on other platforms.)

## Results

Full writeup, per-field breakdown, improved/failure examples, and latency numbers in
[`results/EVAL_REPORT.md`](results/EVAL_REPORT.md). Headline overall-F1 numbers (n=21
synthetic / n=100 SROIE test examples):

| | baseline (zero-shot) | fine-tuned (LoRA) |
|---|---|---|
| Synthetic test set | 0.9268 | 0.9184 (~flat) |
| SROIE test set | 0.6033 | **0.8533** (+0.25) |

Fine-tuning barely moved clean synthetic-text performance (already strong zero-shot) but
produced a large, real improvement on noisy real-world OCR text — see the report for why.
