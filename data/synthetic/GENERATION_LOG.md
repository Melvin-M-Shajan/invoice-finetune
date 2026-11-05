# Synthetic Data Generation Log

**Generator model:** `openai/gpt-oss-20b` served via the Groq API (fast inference host).

I didn't have an OpenAI or Anthropic key at build time. The build went
through two intermediate providers before landing here — recorded for transparency:

1. Started with a Gemini API key. `gemini-2.5-flash`/`gemini-2.5-flash-lite` turned out to
   be capped at **20 requests/day** on this key's free tier
   (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, confirmed from a live 429
   response's quota details) — nowhere near enough for ~1000 calls.
2. Switched to `gemma-4-26b-a4b-it` (also via the Gemini API, a separate quota bucket with
   real throughput) — but at ~30-50s/call, generating 1000 examples would take ~10-14
   hours.
3. I then switched to a Groq API key to speed this up. Groq's hosted
   `openai/gpt-oss-20b` responds in well under a second (5/5 and then several more rapid
   test calls all succeeded with no rate-limit errors), so generation switched to Groq and
   the 1000-example target became practical again (~15-20 min total).
   The first 25 examples generated via the Gemma path were kept (generation is resumable —
   `generate_synthetic` only tops up the remaining count).

`openai/gpt-oss-20b` returns its chain-of-thought in a separate `reasoning` field from the
final `content` field (OpenAI-compatible chat-completions response shape) — only `content`
is used as the rendered invoice text; see `_call_groq` in `src/data/prepare_dataset.py`.


## Method

Ground truth JSON is generated first in Python (`_make_ground_truth` in
`src/data/prepare_dataset.py`): random vendor name (company or individual style), 1-6 line
items with random descriptions/quantities/unit prices, a computed `subtotal`, an optional
tax line (~70% of examples) with a randomly chosen realistic tax rate, and `total =
subtotal + tax`. This guarantees ground truth is correct by construction — the generator
model is only asked to *render* that already-correct JSON as free-form invoice text, never
asked to invent or compute numbers itself.

Varied per example: number of line items (1-6), presence/absence of tax, currency
symbol ($, €, £), date display format (5 formats), whether subtotal is shown explicitly
(~85% of examples), vendor name style (~70% company / ~30% individual).

## Exact prompt template

The following template (from `_RENDER_PROMPT_TEMPLATE` in `src/data/prepare_dataset.py`) is
filled in per-example with that example's `currency_symbol`, `invoice_date_display`, and the
ground-truth JSON (with internal-only helper keys stripped), then sent as a single-turn user
message to `POST https://api.groq.com/openai/v1/chat/completions` with
`"model": "openai/gpt-oss-20b"` (plain REST call via `requests`, not an SDK — see
`_call_groq` in `src/data/prepare_dataset.py`):

```
You are generating realistic-looking invoice OCR/text-layer text for a training dataset. You will be given a ground-truth JSON object describing an invoice's fields. Render this JSON as free-form invoice text that a person reading the raw text of a PDF invoice would see. Follow these rules strictly:

1. Every value in the JSON must appear somewhere in the rendered text, expressed using the given currency symbol "{currency_symbol}" and the given display date format "{invoice_date_display}" for the invoice date (do not invent a different date).
2. Vary the layout and wording naturally (e.g. "Invoice #", "Inv No.", "Bill To" vs no "Bill To" section, table-style line items vs prose-style line items, "Subtotal" shown or omitted per the JSON, "Tax"/"VAT"/"Sales Tax" wording).
3. Do NOT include the JSON itself, field names like "vendor" or "line_items", or any commentary/markdown — output ONLY the plain invoice text.
4. Do NOT change, recompute, round differently, or omit any numeric value from the JSON.

Ground truth JSON:
{ground_truth_json}

Output only the rendered invoice text.
```

## Deduplication

Before finalizing the split, examples are deduplicated on a hash of
`vendor + total + line_item_count` (`_hash_key` in `src/data/prepare_dataset.py`), applied
both incrementally during generation (to avoid wasting API calls on near-duplicates) and
again in `split_and_write` as a final pass.

## Run record

Total requested: 1000 (`split_and_write` reserves 10% of the synthetic
pool as held-out test — exactly 100 at 1000, and scales down correctly if the pool is
smaller, see its docstring in `src/data/prepare_dataset.py`).

**Stopped early at 207.** Even with Groq's fast responses, Groq's free-tier
**8000 tokens/minute** cap on `openai/gpt-oss-20b` throttled sustained throughput (every
call's prompt + reasoning + completion tokens count against it; the retry loop backs off
and retries on every 429, so it never fails, just slows down). After watching it run for a
while and seeing the pace, I decided 207 was enough for this build rather than
waiting out the full 1000. `split_and_write` reserves 10% of whatever pool it gets, so this
produced 21 synthetic test examples instead of 100 (≈635 lines total run 90/10-split for
synthetic train/val, combined with the SROIE pool — see the actual counts below). Expect
noisier metrics, especially line-item precision/recall, than the full 1000-example design
would give — noted again in `README.md` and `results/EVAL_REPORT.md`.

- Command: `python -m src.data.prepare_dataset --n-synthetic 1000` (generation stopped
  manually at 207, then `python -m src.data.prepare_dataset --skip-generation` run to
  produce the final split from what had been generated)
- Actual count generated: 207
- Date run: 2026-09-15
- Any examples that failed after retries (count): 0 observed — all 429s recovered via
  retry; no example was permanently dropped
- Final split: train=617 (167 synthetic + 450 SROIE), val=69 (19 synthetic + 50 SROIE),
  test_synthetic=21, test_sroie=100
