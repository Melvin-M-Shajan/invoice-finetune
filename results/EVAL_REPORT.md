# Evaluation Report — Invoice Extraction LoRA Fine-Tune

_Status: DRAFT — placeholders below are filled in only from real eval runs (baseline runs
BEFORE any fine-tuning happens, per PRD §7.2 and the project's critical rule against
fabricated numbers). Do not treat any number in this file as final until this notice is
removed._

Base model: `Qwen/Qwen2.5-1.5B-Instruct` (see [README.md](../README.md) Deviations section
for why, not Llama-3-8B-Instruct / Mistral-7B-Instruct-v0.3).

## Results table (PRD §7.4)

| Field | baseline_synthetic | finetuned_synthetic | baseline_sroie | finetuned_sroie |
|---|---|---|---|---|
| vendor | TBD | TBD | TBD | TBD |
| invoice_number | TBD | TBD | n/a | n/a |
| invoice_date | TBD | TBD | TBD | TBD |
| line_items precision | TBD | TBD | n/a | n/a |
| line_items recall | TBD | TBD | n/a | n/a |
| subtotal | TBD | TBD | n/a | n/a |
| tax | TBD | TBD | n/a | n/a |
| total | TBD | TBD | TBD | TBD |
| **overall F1** | TBD | TBD | TBD | TBD |

(`n/a` = SROIE ground truth only covers vendor/invoice_date/total, §5.1.)

## Improved examples (fine-tuned beat baseline)

_TBD — 3 examples filled in after both eval runs complete._

## Remaining failure cases (fine-tuned still wrong)

_TBD — 2 documented failure modes filled in after both eval runs complete. Reported
honestly per project rule: do not hide failure cases._

## Inference latency

| | mean seconds/example (batch size 1) |
|---|---|
| baseline (zero-shot) | TBD |
| fine-tuned | TBD |

Hardware: NVIDIA GeForce GTX 1650 (4GB VRAM), 4-bit QLoRA inference.

## Dataset size caveat

Synthetic test set is 21 examples (not the PRD's specified 100) — generation was stopped
early at 207 total synthetic examples due to Groq free-tier throughput limits (see
README/GENERATION_LOG deviation notes). Treat percentage-point differences as noisier/
lower-confidence than a 100-example test set would give.

