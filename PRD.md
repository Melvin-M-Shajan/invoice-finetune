# PRD — Fine-Tuned Invoice/Document Extraction Model

Status: Spec — not yet built. This document is written so a coding agent can implement the
project end-to-end without further clarification. Every interface, schema, and hyperparameter
below is a concrete default — an implementing agent may only deviate from a default when it
hits a documented blocker (see §13 Risks), and must record the deviation and why in the repo
README.

## 1. Purpose
Prove hands-on ML engineering (fine-tuning, evaluation, deployment) to back an "AI/ML
Engineer" resume positioning, using LoRA fine-tuning on a real, measurable task.

## 2. Problem
Off-the-shelf LLMs used zero-shot for structured extraction (the approach used in the
NCompass invoice assistant) are inconsistent on field formatting, hallucinate totals, and
don't reliably follow a fixed JSON schema. A LoRA-fine-tuned model should be more accurate
and cheaper to run at inference time than prompting a larger general-purpose model.

## 3. Goal
Fine-tune a small open-source LLM with LoRA to extract structured fields from invoice
text, and produce a documented, reproducible before/after accuracy comparison against the
base model's zero-shot performance.

## 4. Scope

**In scope**
- One document type: invoices only (do not add receipts or POs — out of scope, see below).
- Base model: `meta-llama/Meta-Llama-3-8B-Instruct` (Hugging Face Hub). If gated-access
  approval is not obtained within 1 day, fall back to `mistralai/Mistral-7B-Instruct-v0.3`
  — document which one was actually used in the README, first line.
- Fine-tuning method: LoRA via Hugging Face `peft` + `trl` (`SFTTrainer`).
- Dataset: SROIE 2019 (ICDAR Robust Reading Challenge, Task 3 — Key Information Extraction)
  as the base, supplemented with synthetically generated invoices (see §5.3).
- Evaluation: field-level exact-match and normalized-match accuracy + F1, base model
  zero-shot vs. fine-tuned model, on a held-out test split.
- Training: local GPU if available (≥16GB VRAM), otherwise AWS SageMaker training job
  (`ml.g5.xlarge` — 1x A10G, 24GB VRAM) using QLoRA (4-bit) if VRAM is the constraint.
- Inference/hosting for the demo: local `transformers` inference is sufficient; SageMaker
  real-time endpoint or Bedrock custom model import is optional/stretch, not required for
  "done."

**Out of scope (do not build these)**
- Multi-document-type generalization (contracts, purchase orders, etc.).
- A custom OCR/vision pipeline — assume text-layer PDFs or pre-OCR'd text as input; SROIE
  already ships OCR text boxes.
- Multi-language support (English only).
- Real-time serving SLAs, autoscaling, or load testing.
- A web frontend beyond the single demo script described in §8.

## 5. Data Pipeline

### 5.1 SROIE dataset
- Source: SROIE 2019 (search "ICDAR2019 SROIE dataset"; several mirrors exist on
  Hugging Face Datasets and Kaggle — use the Hugging Face Datasets version if available for
  reproducibility, e.g. `darentang/sroie` or equivalent; verify license permits use in a
  public portfolio repo before publishing, and record the exact source URL used in the
  README's Data section).
- Raw format: one folder per document with `.jpg` (receipt image), `.txt` (OCR text boxes,
  line-by-line: `x1,y1,x2,y2,x3,y3,x4,y4,text`), and `.json` (ground-truth key-value: at
  minimum `company`, `date`, `address`, `total`).
- SROIE's native ground truth only covers 4 fields (`company`, `date`, `address`, `total`)
  and is receipt-shaped, not itemized-invoice-shaped. Because this project's target schema
  (§5.2) needs line items, tax, invoice number — SROIE is used only for `vendor` (←
  `company`), `date`, and `total` ground truth, plus as a source of realistic OCR'd raw text
  to prompt-condition synthetic generation. Line items and invoice numbers come entirely
  from the synthetic set (§5.3). State this limitation plainly in the README and in the PRD
  origin — do not claim SROIE provides line-item ground truth.

### 5.2 Target extraction schema
Every model output (base and fine-tuned) must be validated against this JSON schema before
scoring. A response that fails to parse as valid JSON against this schema counts as 0 for
every field it contains (do not silently drop it from the eval set).

```json
{
  "vendor": "string",
  "invoice_number": "string | null",
  "invoice_date": "string (YYYY-MM-DD) | null",
  "line_items": [
    {
      "description": "string",
      "quantity": "number",
      "unit_price": "number",
      "line_total": "number"
    }
  ],
  "subtotal": "number | null",
  "tax": "number | null",
  "total": "number"
}
```

### 5.3 Synthetic data generation
- Target: 800 synthetic invoices for training, 100 for validation, 100 for test (test set
  must be fully held out — never touched during training or hyperparameter selection).
- Generation method: use a strong instruction model (GPT-4o-mini, Claude Haiku, or
  equivalent low-cost API model — record which one and the exact prompt used in
  `data/synthetic/GENERATION_LOG.md`) to generate `(invoice_text, ground_truth_json)` pairs.
  Generate the ground-truth JSON first (random vendor names, 1-6 line items, realistic
  quantities/prices, computed subtotal/tax/total), then prompt the generator model to render
  that JSON as free-form invoice text (varying layout/wording), so ground truth is
  guaranteed correct by construction — do not generate free text first and hand-label it.
- Vary: number of line items (1-6), presence/absence of tax, currency symbol style, date
  format, whether subtotal is shown explicitly, vendor name style (individual vs. company).
- Store as JSONL: one line per example, `{"invoice_text": "...", "ground_truth":
  {...schema from 5.2...}}`.
- Deduplicate near-identical vendor/line-item combinations (hash on
  vendor+total+line_item_count) before finalizing the split.

### 5.4 Train/val/test split
- 800 synthetic (train) + up to 500 SROIE examples usable for the 3 SROIE-covered fields
  (train, partial-label — only vendor/date/total scored when SROIE-sourced) : split 90/10
  into train/val within each source, keep sources balanced across the split.
- 100 synthetic (test, full schema) + 100 SROIE (test, partial schema, vendor/date/total
  only) — report these two test subsets' metrics **separately**, never blended into one
  number, since they test different things (full extraction vs. 3-field extraction).

### 5.5 Preprocessing script — `src/data/prepare_dataset.py`
Function signatures (implement exactly these, additional private helpers are fine):

```python
def load_sroie(raw_dir: str) -> list[dict]:
    """Returns list of {"invoice_text": str, "ground_truth": {partial schema}} from raw SROIE."""

def generate_synthetic(n: int, generator_model: str, out_path: str) -> None:
    """Calls the generator API n times, writes JSONL to out_path. Must be resumable —
    if out_path already has k lines, only generate n - k more."""

def build_training_example(invoice_text: str, ground_truth: dict) -> dict:
    """Formats one example into the instruction-tuning chat format (see §6.3)."""

def split_and_write(examples: list[dict], out_dir: str, seed: int = 42) -> None:
    """Writes train.jsonl, val.jsonl, test.jsonl per §5.4 split rules."""
```

## 6. Model & Training

### 6.1 Base model
`meta-llama/Meta-Llama-3-8B-Instruct` (fallback `mistralai/Mistral-7B-Instruct-v0.3` — see
§4).

### 6.2 LoRA configuration
Exact starting hyperparameters (tune only if the baseline run in §7 shows underfitting —
record any change and why):

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

Load base model in 4-bit (QLoRA) via `bitsandbytes` if VRAM < 24GB:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
```

### 6.3 Prompt / instruction format
Every training example is formatted as a chat-style instruction. System prompt is fixed and
identical at train and inference time (do not let it drift):

```
SYSTEM:
You are an invoice data extraction assistant. Given raw invoice text, extract the
following fields and return ONLY a single JSON object matching this schema, with no
extra commentary: vendor, invoice_number, invoice_date (YYYY-MM-DD), line_items
(array of {description, quantity, unit_price, line_total}), subtotal, tax, total.
Use null for fields not present in the text.

USER:
{invoice_text}

ASSISTANT:
{json.dumps(ground_truth, ensure_ascii=False)}
```

### 6.4 Training script — `src/train/train_lora.py`
```python
def train(
    train_path: str,
    val_path: str,
    base_model: str,
    output_dir: str,
    num_epochs: int = 3,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    seed: int = 42,
) -> str:
    """Runs SFTTrainer with the lora_config from §6.2, the prompt format from §6.3.
    Saves LoRA adapter weights (not merged) to output_dir. Returns output_dir.
    Logs train/val loss every 10 steps to a local file AND to Weights & Biases if
    WANDB_API_KEY is set (optional — do not fail if it's not set, just skip W&B)."""
```
Entry point: `python -m src.train.train_lora --train data/processed/train.jsonl --val
data/processed/val.jsonl --output-dir models/invoice-lora-v1`.

### 6.5 AWS SageMaker training job (if local GPU unavailable)
- Use a SageMaker HuggingFace Estimator, entry point `src/train/train_lora.py`.
- Instance: `ml.g5.xlarge` (1x A10G 24GB) or `ml.g5.2xlarge` if more headroom needed.
- `requirements.txt` pinned versions (see §9).
- Upload `data/processed/` to S3 first (`s3://<bucket>/invoice-finetune/data/`); pass as
  SageMaker channel inputs `train` and `val`.
- Save the exact `estimator.fit()` call and instance config used in
  `infra/sagemaker_train.py` so it's reproducible, including the S3 output path for the
  trained adapter.

## 7. Evaluation

### 7.1 Metrics — `src/eval/metrics.py`
```python
def exact_match(pred_value, true_value) -> bool:
    """Case-insensitive, whitespace-normalized string equality. For numbers, compare after
    parsing to float."""

def normalized_match(pred_value, true_value, field_type: str) -> bool:
    """field_type in {"string", "date", "number"}. Dates: parse both to ISO YYYY-MM-DD
    before comparing (accept common formats). Numbers: match if abs(pred - true) < 0.01.
    Falls back to exact_match for anything that fails to parse."""

def score_example(pred_json: dict | None, true_json: dict) -> dict:
    """Returns per-field bool correctness, e.g. {"vendor": True, "invoice_number": False,
    ...}. If pred_json is None (failed to parse as valid JSON), every field is False.
    Line items are scored as a set: a predicted line item counts correct if there exists an
    unmatched true line item with normalized_match True on description AND quantity AND
    unit_price; report line_items_precision and line_items_recall separately from the
    scalar fields."""

def aggregate(all_scores: list[dict]) -> dict:
    """Returns per-field accuracy (mean of bool correctness across examples), plus overall
    F1 computed as 2PR/(P+R) where P = mean per-field precision, R = mean per-field
    recall, treating each field as a binary classification (present+correct vs. not)."""
```

### 7.2 Baseline run (must run BEFORE any fine-tuning, do not skip)
`src/eval/run_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --mode zero-shot --test
data/processed/test_synthetic.jsonl --out results/baseline_synthetic.json`, and again against
`test_sroie.jsonl`. Zero-shot means: use the exact same system prompt from §6.3, no
few-shot examples, base model weights only (no adapter loaded).

### 7.3 Fine-tuned eval
Same script, `--mode finetuned --adapter models/invoice-lora-v1`, against both test subsets.

### 7.4 Report — `results/EVAL_REPORT.md`
Must contain, at minimum:
- A markdown table: rows = fields (vendor, invoice_number, invoice_date, line_items P/R,
  subtotal, tax, total, overall F1), columns = (baseline_synthetic, finetuned_synthetic,
  baseline_sroie, finetuned_sroie).
- 3 example predictions where the fine-tuned model improved over baseline (input text,
  both outputs, ground truth).
- 2 example predictions where the fine-tuned model is still wrong (documented failure
  modes — e.g., multi-currency, handwritten-style OCR noise, ambiguous tax lines). Do not
  omit or hide failure cases.
- Inference latency: mean seconds/example for base model vs. fine-tuned model on the same
  hardware, single-example batch size.

## 8. Demo — `src/demo/cli.py`
```python
def extract(invoice_text: str, adapter_path: str = "models/invoice-lora-v1") -> dict:
    """Loads base model + LoRA adapter, runs the prompt from §6.3, parses and returns the
    JSON dict. Raises a clear error (not a silent None) if the model output fails to
    parse as valid JSON against the §5.2 schema."""
```
CLI usage: `python -m src.demo.cli --input path/to/invoice.txt` prints the extracted JSON to
stdout. A Gradio wrapper (`src/demo/app.py`, textbox in → JSON out) is optional/stretch.

## 9. Environment & Dependencies
`requirements.txt` (pin exact versions actually used at build time — these are minimum
versions to start from):
```
transformers>=4.42
peft>=0.11
trl>=0.9
bitsandbytes>=0.43
accelerate>=0.31
datasets>=2.19
torch>=2.3
sagemaker>=2.220  # only if using SageMaker training
boto3>=1.34        # only if using SageMaker/S3
gradio>=4.36        # only if building the optional demo UI
```
`.env` variables needed (document all of these in `.env.example`, never commit real values):
- `HF_TOKEN` — for gated Llama 3 access.
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` — for synthetic data generation (§5.3).
- `WANDB_API_KEY` — optional, training logging.
- `AWS_PROFILE` / `AWS_REGION` — only if using SageMaker.

## 10. Repository Structure
```
invoice-finetune/
  README.md                  # setup, results summary, how to reproduce
  requirements.txt
  .env.example
  data/
    raw/sroie/                # gitignored — download script fetches this
    synthetic/
      GENERATION_LOG.md       # exact prompt + model used for synthetic generation
    processed/
      train.jsonl
      val.jsonl
      test_synthetic.jsonl
      test_sroie.jsonl
  src/
    data/prepare_dataset.py
    train/train_lora.py
    eval/metrics.py
    eval/run_eval.py
    demo/cli.py
    demo/app.py               # optional
  infra/
    sagemaker_train.py        # optional, only if SageMaker used
  models/                      # gitignored — LoRA adapters saved here locally
  results/
    baseline_synthetic.json
    finetuned_synthetic.json
    baseline_sroie.json
    finetuned_sroie.json
    EVAL_REPORT.md
  tests/
    test_metrics.py
    test_prepare_dataset.py
```

## 11. Testing Plan
- Unit tests for `src/eval/metrics.py` (§7.1 functions) with hand-constructed
  pred/true pairs covering: exact match, normalized date match, numeric tolerance match,
  line-item set matching with partial overlap, malformed JSON input.
- Unit test for `build_training_example` — verify output format exactly matches §6.3.
- A smoke test that runs `extract()` (§8) against one fixture invoice and asserts the
  output parses as valid JSON against the §5.2 schema (does not assert correctness, just
  that the pipeline runs end-to-end).

## 12. Milestones (Definition of Done per milestone)
1. **Data pipeline** — `prepare_dataset.py` runs end-to-end, produces all 4 processed
   JSONL files matching §5.4 split sizes, `GENERATION_LOG.md` exists and documents the
   generation prompt/model. DoD: `tests/test_prepare_dataset.py` passes.
2. **Baseline eval** — `results/baseline_synthetic.json` and `results/baseline_sroie.json`
   exist, generated by §7.2, before any training run happens. DoD: files exist with
   per-field scores for every test example.
3. **Fine-tuning** — `models/invoice-lora-v1/` contains a trained adapter; training loss
   curve logged. DoD: adapter loads successfully in `extract()` (§8) without error.
4. **Fine-tuned eval** — `results/finetuned_synthetic.json` and
   `results/finetuned_sroie.json` exist per §7.3. DoD: same format as baseline, directly
   comparable.
5. **Report + demo** — `results/EVAL_REPORT.md` complete per §7.4; `src/demo/cli.py`
   works on a sample invoice. DoD: README's "Results" section links to EVAL_REPORT.md and
   states the actual final F1 numbers (never draft resume bullets with placeholder
   numbers — always copy the real ones from this report).

## 13. Risks & Mitigations
- **Llama 3 8B gated access delay** → fall back to Mistral 7B Instruct v0.3 (§4), document
  the switch.
- **LoRA shows only marginal improvement on a small dataset** → this is a valid, reportable
  outcome — do not inflate or cherry-pick metrics; report the real number and discuss why
  (e.g., zero-shot baseline for JSON-schema-following may already be strong for a modern
  instruct model, so the more interesting metric may be line-item extraction specifically,
  where zero-shot is typically much weaker).
- **SROIE license terms** — confirm redistribution/use terms before making the repo public;
  if unclear, keep `data/raw/sroie/` gitignored and only publish `data/synthetic/` and the
  processed JSONL derived from synthetic data, noting in the README that SROIE must be
  downloaded separately by anyone reproducing the SROIE-eval numbers.
- **VRAM constraints** — use QLoRA (4-bit, §6.2) and/or reduce `per_device_batch_size` with
  higher `gradient_accumulation_steps` before reaching for a bigger instance.
