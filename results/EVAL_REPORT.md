# Evaluation Report — Invoice Extraction LoRA Fine-Tune

All numbers below come from actual eval runs (`src/eval/run_eval.py`) executed against the
held-out test sets — the baseline (zero-shot) runs happened *before* any fine-tuning. Raw per-example outputs are in
`results/baseline_synthetic.json`, `results/finetuned_synthetic.json`,
`results/baseline_sroie.json`, `results/finetuned_sroie.json`.

Base model: `Qwen/Qwen2.5-1.5B-Instruct` (see [README.md](../README.md) Design decisions section
for why, not Llama-3-8B-Instruct / Mistral-7B-Instruct-v0.3).

## Results table

| Field | baseline_synthetic | finetuned_synthetic | baseline_sroie | finetuned_sroie |
|---|---|---|---|---|
| vendor | 0.9524 | 1.0000 | 0.5800 | 0.6900 |
| invoice_number | 0.9524 | 0.9048 | n/a | n/a |
| invoice_date | 0.9524 | 0.9524 | 0.8500 | 0.9400 |
| line_items precision | 0.8714 | 0.8571 | n/a | n/a |
| line_items recall | 0.8651 | 0.8571 | n/a | n/a |
| subtotal | 0.8571 | 0.8571 | n/a | n/a |
| tax | 0.9524 | 0.9048 | n/a | n/a |
| total | 0.9524 | 0.9524 | 0.3800 | 0.9300 |
| **overall F1** | **0.9268** | **0.9184** | **0.6033** | **0.8533** |

(`n/a` = SROIE ground truth only covers vendor/invoice_date/total; n=21 for the
synthetic test set, n=100 for SROIE.)

**Headline result:** fine-tuning barely moved the needle on clean synthetic text
(0.9268 → 0.9184, essentially flat/slightly down) but produced a large real improvement
on messy real-world OCR text (SROIE: 0.6033 → 0.8533, +0.25), driven mostly by `total`
(0.38 → 0.93) and `vendor` (0.58 → 0.69). This is a known risk: a modern instruct model's zero-shot JSON-schema-following can already be strong
on clean text, so the more interesting signal is how fine-tuning helps on the harder,
noisier distribution the base model wasn't specifically prompted/tuned for.

## Improved examples (fine-tuned beat baseline)

All three are from the SROIE test set, where the effect was clearest.

**1. SROIE test idx 30** — baseline invented a hallucinated line item and got every field
wrong; fine-tuned got all 3 scored fields right.
- Input (truncated): `HON HWA HARDWARE TRADING ... TAX INVOICE CB#: 69379 11/02/2017 ... 3/4" ALUMINIUM ROD 5.80 19.60 ... Total 19.60 ... Total Inclusive GST: 19.60`
- Ground truth: `{"vendor": "HON HWA HARDWARE TRADING", "invoice_date": "2017-02-11", "total": 19.6}`
- Baseline: `{"vendor": "HON HWA HARDWARE", "invoice_number": "C1-0", "invoice_date": "2017-11-02", "line_items": [{"description": "3/4\" ALUMINIUM ROD", "quantity": 5, "unit_price": 1.38, "line_total": 7.89}], "subtotal": 19.6, "tax": 1.11, "total": 20.71}` — wrong vendor, swapped day/month in the date, hallucinated a line-item quantity/price that doesn't multiply out to its own line_total, wrong total.
- Fine-tuned: `{"vendor": "HON HWA HARDWARE TRADING", "invoice_date": "2017-02-11", "total": 19.6}` — all correct.

**2. SROIE test idx 49** — baseline failed to produce valid JSON at all (scored 0 on every
field); fine-tuned got all 3 right.
- Input (truncated): `SYARIKAT PERNIAGAAN GIN KEE (81109-A) ... SIMPLIFIED TAX INVOICE ... Date: 12/02/2018 ... 110MM X 5.8M UNDERGROUND PIPE Total Qty: 3 270.30 ... Total Sales (Inclusive of GST) : 270.30`
- Ground truth: `{"vendor": "SYARIKAT PERNIAGAAN GIN KEE", "invoice_date": "2018-02-12", "total": 270.3}`
- Baseline: failed to parse as JSON (`prediction: None`).
- Fine-tuned: `{"vendor": "SYARIKAT PERNIAGAAN GIN KEE", "invoice_date": "2018-02-12", "total": 270.3}` — all correct.

**3. SROIE test idx 55** — same pattern: baseline produced unparseable output, fine-tuned
extracted cleanly from a receipt with a long, noisy itemized list.
- Input (truncated): `THAI DELICIOUS RESTAURANT ... Bill No 01-36870 ... Date 02/02/2017 ... Description Qty Price Amount 06S BRAISED 20.00 20.00 PORK LEG SMALL ...` (8 line items follow)
- Ground truth: `{"vendor": "THAI DELICIOUS RESTAURANT", "invoice_date": "2017-02-02", "total": 92.8}`
- Baseline: failed to parse as JSON (`prediction: None`).
- Fine-tuned: `{"vendor": "THAI DELICIOUS RESTAURANT", "invoice_date": "2017-02-02", "total": 92.8}` — all correct.

Across all of `test_sroie.jsonl`, 53/100 examples improved (more fields correct after
fine-tuning than before); this is a large share of why SROIE's overall F1 jumped by 0.25.

## Remaining failure cases (fine-tuned still wrong)



**1. Ambiguous subtotal — model fills in a value that isn't actually in the text
(synthetic test idx 12).**
- Input: an invoice listing 4 line items and a `Total: $7181.89` line, with **no**
  `Subtotal` line printed anywhere in the text (by design — the generator varies whether subtotal is shown explicitly, and this example was generated with it hidden).
- Ground truth: `subtotal: null, tax: null, total: 7181.89` (correct per the system
  prompt's instruction to use `null` for fields not present in the text).
- Fine-tuned prediction: `subtotal: 7181.89` (copied from `total`) instead of `null` —
  the model infers/back-fills a subtotal instead of recognizing it was never printed.
  Everything else in the prediction is correct.
- The model conflates "value is mathematically inferable" with "value was stated in text."

**2. Numeric total mismatch on a receipt with multiple candidate totals
(SROIE test idx 5).**
- Input (truncated): `Subway#42210-0 ... Qty Size Item Price ... Bag Fee RM0.20 - - (MY) EDV
  Meal RM11.90 ... (MY) EDV Meal RM11.90 ... Cash Rounding -RM0.01 Total (Eat In) RM24.00`
- Ground truth: `total: 22.65`.
- Fine-tuned prediction: `total: 24.0` — which is *exactly* the figure printed next to
  "Total (Eat In)" in the OCR text. The receipt has several candidate numbers (two
  RM11.90 combo items, a RM0.20 bag fee, a -RM0.01 rounding adjustment, and the RM24.00
  "Total (Eat In)" line) and the ground-truth 22.65 doesn't obviously match any simple
  combination of the visible line items either — this looks like either an OCR-text
  reconstruction gap (the text file used for this eval doesn't preserve every printed
  token/line perfectly) or a SROIE annotation quirk, not a case where the model's
  arithmetic reasoning is clearly at fault. Documented rather than swept under the rug.

## Inference latency

| | mean seconds/example (batch size 1) |
|---|---|
| baseline zero-shot, synthetic | 37.62 |
| fine-tuned, synthetic | 27.70 |
| baseline zero-shot, SROIE | 24.77 |
| fine-tuned, SROIE | 6.34 |

Fine-tuned is faster as well as more accurate on SROIE — the base model tends to ramble
(extra commentary, retries at matching the schema) before settling on an answer or giving
up unparseable, while the fine-tuned model has learned to emit the JSON object directly
and stop, so generation terminates much sooner on average.

Hardware: NVIDIA GeForce GTX 1650 (4GB VRAM), 4-bit QLoRA inference, greedy decoding
(`do_sample=False`), single-example batches.

## Dataset size caveat

Synthetic test set is 21 examples (the original target was 100) — generation was stopped
at 207 total synthetic examples partway through a Groq-throughput-
limited run (see README and GENERATION_LOG). Treat the synthetic-column
percentage-point differences as noisier/lower-confidence than a 100-example test set would
give; the SROIE column (n=100, unaffected by this deviation) is the more statistically
solid of the two results.
