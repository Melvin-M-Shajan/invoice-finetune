"""Data pipeline for invoice extraction fine-tuning (PRD §5).

CLI entry point (see bottom of file / README "Reproduce" section):
    python -m src.data.prepare_dataset --raw-dir data/raw/sroie \
        --synthetic-out data/synthetic/synthetic_raw.jsonl \
        --processed-dir data/processed --n-synthetic 1000
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from dateutil import parser as dateutil_parser
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an invoice data extraction assistant. Given raw invoice text, extract the "
    "following fields and return ONLY a single JSON object matching this schema, with no "
    "extra commentary: vendor, invoice_number, invoice_date (YYYY-MM-DD), line_items "
    "(array of {description, quantity, unit_price, line_total}), subtotal, tax, total. "
    "Use null for fields not present in the text."
)

# ---------------------------------------------------------------------------
# §5.1 SROIE loading
# ---------------------------------------------------------------------------


def _parse_sroie_date(raw: str):
    try:
        return dateutil_parser.parse(raw, fuzzy=True, dayfirst=True).date().isoformat()
    except (ValueError, OverflowError, TypeError):
        return raw


def _parse_sroie_total(raw):
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = str(raw).replace(",", "").replace("$", "").replace("RM", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _sroie_bio_to_entities(words: list[str], tag_ids: list[int], tag_names: list[str]) -> dict:
    """Reconstructs company/date/total spans from the HF darentang/sroie BIO tagging
    scheme (labels: O, B-/I-{COMPANY,DATE,ADDRESS,TOTAL})."""
    entities: dict[str, list[str]] = {"COMPANY": [], "DATE": [], "ADDRESS": [], "TOTAL": []}
    current_type = None
    current_tokens: list[str] = []

    def flush():
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            entities[current_type].append(" ".join(current_tokens))
        current_type, current_tokens = None, []

    for word, tag_id in zip(words, tag_ids):
        tag = tag_names[tag_id]
        if tag == "O":
            flush()
            continue
        prefix, _, entity_type = tag.partition("-")
        if prefix == "B" or entity_type != current_type:
            flush()
            current_type, current_tokens = entity_type, [word]
        else:  # "I" continuing the same span
            current_tokens.append(word)
    flush()

    return {
        "company": entities["COMPANY"][0] if entities["COMPANY"] else None,
        "date": entities["DATE"][0] if entities["DATE"] else None,
        "address": entities["ADDRESS"][0] if entities["ADDRESS"] else None,
        "total": entities["TOTAL"][0] if entities["TOTAL"] else None,
    }


# Confirmed via https://datasets-server.huggingface.co/info?dataset=darentang/sroie —
# used as a fallback since the parquet mirror may not preserve the HF ClassLabel names.
_SROIE_NER_TAG_NAMES = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]


def download_sroie(raw_dir: str, hf_dataset: str = "darentang/sroie") -> None:
    """Materializes the HF `darentang/sroie` dataset into the PRD §5.1 raw format
    (one <id>.txt of OCR text + one <id>.json of ground truth per document) under
    raw_dir, so load_sroie() can read it generically. Combines both HF splits (train
    + test) into one flat pool — this repo's own train/val/test split (§5.4) is
    computed fresh by split_and_write, so the original HF split boundary is not
    load-bearing here. No-ops if raw_dir already has any .json files."""
    raw_path = Path(raw_dir)
    if raw_path.exists() and any(raw_path.rglob("*.json")):
        return

    import datasets as hf_datasets

    raw_path.mkdir(parents=True, exist_ok=True)
    # `datasets` >=4 dropped support for legacy loading-script datasets (darentang/sroie
    # ships one); load the HF-maintained auto-converted parquet mirror directly instead.
    data_files = {
        "train": f"hf://datasets/{hf_dataset}@refs%2Fconvert%2Fparquet/sroie/train/0000.parquet",
        "test": f"hf://datasets/{hf_dataset}@refs%2Fconvert%2Fparquet/sroie/test/0000.parquet",
    }
    dataset_dict = hf_datasets.load_dataset("parquet", data_files=data_files)

    for split_name, split in dataset_dict.items():
        tag_feature = split.features["ner_tags"].feature
        tag_names = tag_feature.names if hasattr(tag_feature, "names") else _SROIE_NER_TAG_NAMES
        for row in split:
            doc_id = f"{split_name}_{row['id']}"
            words = row["words"]
            entities = _sroie_bio_to_entities(words, row["ner_tags"], tag_names)

            (raw_path / f"{doc_id}.txt").write_text(" ".join(words), encoding="utf-8")
            (raw_path / f"{doc_id}.json").write_text(json.dumps(entities, ensure_ascii=False), encoding="utf-8")

    print(f"download_sroie: wrote {sum(1 for _ in raw_path.glob('*.json'))} documents to {raw_dir}")


def load_sroie(raw_dir: str) -> list[dict]:
    """Returns list of {"invoice_text": str, "ground_truth": {partial schema}} from raw
    SROIE. Expects raw_dir laid out per PRD §5.1: one subfolder per document containing
    <id>.txt (OCR line boxes: x1,y1,x2,y2,x3,y3,x4,y4,text) and <id>.json (ground truth:
    company, date, address, total). Only vendor/invoice_date/total are mapped into the
    target schema — SROIE has no line items or invoice number (§5.1 limitation)."""
    examples: list[dict] = []
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return examples

    json_files = sorted(raw_path.rglob("*.json"))
    for json_file in json_files:
        txt_file = json_file.with_suffix(".txt")
        if not txt_file.exists():
            continue
        try:
            gt_raw = json.loads(json_file.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue

        raw_txt = txt_file.read_text(encoding="utf-8", errors="ignore")
        lines = []
        for line in raw_txt.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)
            # PRD §5.1 OCR box format: 8 coords + text. Anything else (e.g. this repo's
            # own HF-materialized plain-text files) is passed through as-is.
            if len(parts) == 9 and all(p.strip().lstrip("-").isdigit() for p in parts[:8]):
                lines.append(parts[8])
            else:
                lines.append(line)
        invoice_text = "\n".join(lines) if lines else raw_txt.strip()
        if not invoice_text:
            continue

        company = gt_raw.get("company")
        date = gt_raw.get("date")
        total = gt_raw.get("total")

        examples.append(
            {
                "invoice_text": invoice_text,
                "ground_truth": {
                    "vendor": company,
                    "invoice_date": _parse_sroie_date(date) if date else None,
                    "total": _parse_sroie_total(total),
                },
                "source": "sroie",
            }
        )
    return examples


# ---------------------------------------------------------------------------
# §5.3 Synthetic data generation
# ---------------------------------------------------------------------------

_VENDOR_COMPANIES = [
    "Acme Supply Co.", "Blue Ridge Hardware", "Cascade Office Solutions", "Delta Freight Lines",
    "Evergreen Landscaping", "Falcon Electronics", "Golden Gate Catering", "Harbor Print Shop",
    "Ironclad Security Services", "Juniper Consulting Group", "Kestrel Cleaning Co.",
    "Lakeside Plumbing", "Maple Grove Bakery", "Northwind Logistics", "Orchid Textiles",
    "Pinnacle IT Services", "Quartz Construction", "Riverside Auto Repair", "Summit Legal Group",
    "Timberline Roofing",
]
_VENDOR_INDIVIDUALS = [
    "John A. Reyes", "Maria Chen", "David O'Brien", "Priya Nair", "Samuel Okafor",
    "Elena Petrova", "Carlos Mendoza", "Grace Lindqvist", "Ahmed Hassan", "Laura Kowalski",
]
_ITEM_DESCRIPTIONS = [
    "Consulting services (hourly)", "Widget A - standard", "Widget B - premium",
    "Office supplies bundle", "Freight and handling", "Laptop stand", "USB-C cable (2m)",
    "Monthly maintenance plan", "Custom logo design", "Replacement filter cartridge",
    "On-site repair labor", "Bulk paper (500 sheets)", "Security camera unit",
    "Landscaping service - weekly", "Catering tray - medium", "Printer toner cartridge",
    "Legal review (per hour)", "Roof inspection", "Plumbing fixture", "Textile roll (per yard)",
]
_CURRENCY_SYMBOLS = ["$", "€", "£"]
_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%d %b %Y"]


def _make_ground_truth(rng: random.Random) -> dict:
    is_company = rng.random() < 0.7
    vendor = rng.choice(_VENDOR_COMPANIES) if is_company else rng.choice(_VENDOR_INDIVIDUALS)

    n_items = rng.randint(1, 6)
    line_items = []
    for _ in range(n_items):
        description = rng.choice(_ITEM_DESCRIPTIONS)
        quantity = rng.randint(1, 20)
        unit_price = round(rng.uniform(1.5, 500.0), 2)
        line_total = round(quantity * unit_price, 2)
        line_items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total,
            }
        )

    subtotal = round(sum(li["line_total"] for li in line_items), 2)
    has_tax = rng.random() < 0.7
    tax_rate = rng.choice([0.05, 0.07, 0.0825, 0.1, 0.15]) if has_tax else 0.0
    tax = round(subtotal * tax_rate, 2) if has_tax else 0.0
    total = round(subtotal + tax, 2)

    invoice_number = f"INV-{rng.randint(1000, 999999)}"
    date_fmt = rng.choice(_DATE_FORMATS)
    days_ago = rng.randint(0, 720)
    date_iso = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days_ago * 86400))
    invoice_date_display = dateutil_parser.parse(date_iso).strftime(date_fmt)

    return {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "invoice_date": date_iso,  # canonical ISO ground truth
        "invoice_date_display": invoice_date_display,  # for prompt rendering only
        "line_items": line_items,
        "subtotal": subtotal if rng.random() < 0.85 else None,  # "whether subtotal is shown"
        "tax": tax if has_tax else None,
        "total": total,
        "_currency_symbol": rng.choice(_CURRENCY_SYMBOLS),
    }


_RENDER_PROMPT_TEMPLATE = """You are generating realistic-looking invoice OCR/text-layer text for a training \
dataset. You will be given a ground-truth JSON object describing an invoice's fields. \
Render this JSON as free-form invoice text that a person reading the raw text of a PDF \
invoice would see. Follow these rules strictly:

1. Every value in the JSON must appear somewhere in the rendered text, expressed using the \
given currency symbol "{currency_symbol}" and the given display date format \
"{invoice_date_display}" for the invoice date (do not invent a different date).
2. Vary the layout and wording naturally (e.g. "Invoice #", "Inv No.", "Bill To" vs no \
"Bill To" section, table-style line items vs prose-style line items, "Subtotal" shown or \
omitted per the JSON, "Tax"/"VAT"/"Sales Tax" wording).
3. Do NOT include the JSON itself, field names like "vendor" or "line_items", or any \
commentary/markdown — output ONLY the plain invoice text.
4. Do NOT change, recompute, round differently, or omit any numeric value from the JSON.

Ground truth JSON:
{ground_truth_json}

Output only the rendered invoice text.
"""


def _hash_key(ground_truth: dict) -> str:
    key = f"{ground_truth.get('vendor')}|{ground_truth.get('total')}|{len(ground_truth.get('line_items') or [])}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_synthetic(n: int, generator_model: str, out_path: str) -> None:
    """Calls the generator API n times, writes JSONL to out_path. Must be resumable —
    if out_path already has k lines, only generate n - k more.

    Ground truth is generated first (random vendor/line items/computed totals) and the
    generator model is only used to render that JSON as free-form invoice text, so ground
    truth is guaranteed correct by construction (§5.3)."""
    import requests

    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY (or GEMINI_API_KEY) not set in environment")
    use_groq = bool(os.environ.get("GROQ_API_KEY"))

    def _call_groq(prompt_text: str) -> str:
        # Groq's OpenAI-compatible chat completions endpoint — orders of magnitude
        # faster than the Gemini free tier for this workload (sub-second vs 30-50s/call).
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": generator_model, "messages": [{"role": "user", "content": prompt_text}]},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code} {resp.text}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {data}")
        # "reasoning" (gpt-oss chain-of-thought) is returned as a separate field from
        # "content" — only content is the rendered invoice text we want.
        return choices[0]["message"].get("content", "")

    def _call_gemini(prompt_text: str) -> str:
        # Plain REST call instead of the google-genai SDK: the SDK's own internal
        # retry-on-429 was observed to silently retry for minutes before ever raising
        # (indistinguishable from a hang), fighting with the rate-limit-aware retry loop
        # below. A direct request with an explicit timeout gives full control.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{generator_model}:generateContent"
        resp = requests.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt_text}]}]},
            timeout=90,  # gemma-4-26b-a4b-it emits a reasoning "thought" pass first; slower than flash tier
        )
        if resp.status_code != 200:
            # Keep the full body (not truncated) — the retryDelay the caller needs to
            # parse for 429s lives in a "details" array well past the first 500 chars.
            raise RuntimeError(f"{resp.status_code} {resp.text}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidates in response: {data}")
        parts = candidates[0].get("content", {}).get("parts") or []
        # Some models (e.g. gemma-4-*-it) emit a reasoning "thought" part before the
        # real answer part — only the non-thought text is the invoice text we want.
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    _call_generator = _call_groq if use_groq else _call_gemini

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    existing = 0
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())

    remaining = n - existing
    if remaining <= 0:
        return

    rng = random.Random(42 + existing)  # offset so resumed runs don't repeat prior examples
    seen_hashes: set[str] = set()
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    seen_hashes.add(_hash_key(row["ground_truth"]))

    # gemini-2.5-flash / gemini-2.5-flash-lite free tier is capped at only 20
    # requests/DAY (GenerateRequestsPerDayPerProjectPerModel-FreeTier) — nowhere near
    # enough for ~1000 calls, confirmed via a 429 response's quota details. Switched to
    # Groq (sub-second responses, 5/5 rapid test calls succeeded) — see README/
    # GENERATION_LOG deviation notes. Still throttle proactively rather than relying only
    # on reactive 429 backoff.
    min_interval = 0.3 if use_groq else 60.0 / 30
    last_call_time = [0.0]

    with out_file.open("a", encoding="utf-8") as f:
        generated = 0
        attempts = 0
        max_attempts = remaining * 3 + 20
        while generated < remaining and attempts < max_attempts:
            attempts += 1
            gt = _make_ground_truth(rng)
            h = _hash_key(gt)
            if h in seen_hashes:
                continue  # §5.3 dedup on vendor+total+line_item_count

            prompt = _RENDER_PROMPT_TEMPLATE.format(
                currency_symbol=gt["_currency_symbol"],
                invoice_date_display=gt["invoice_date_display"],
                ground_truth_json=json.dumps(
                    {k: v for k, v in gt.items() if not k.startswith("_") and k != "invoice_date_display"},
                    ensure_ascii=False,
                ),
            )

            invoice_text = None
            for retry in range(6):
                # Proactive throttling: stay under the API's free-tier requests-per-minute cap.
                wait = min_interval - (time.time() - last_call_time[0])
                if wait > 0:
                    time.sleep(wait)
                try:
                    last_call_time[0] = time.time()
                    invoice_text = _call_generator(prompt).strip()
                    if invoice_text:
                        break
                except Exception as exc:  # noqa: BLE001 - retry on any transient API error
                    exc_str = str(exc)
                    delay_match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", exc_str)
                    sleep_for = float(delay_match.group(1)) + 1.0 if delay_match else min(2 ** retry, 30)
                    print(f"generate_synthetic: retry {retry} after error, sleeping {sleep_for:.1f}s: {exc_str[:200]}", flush=True)
                    time.sleep(sleep_for)
                    if retry == 5:
                        print(f"generate_synthetic: giving up on one example after retries: {exc_str[:200]}", flush=True)
            if not invoice_text:
                continue

            clean_gt = {k: v for k, v in gt.items() if not k.startswith("_") and k != "invoice_date_display"}
            f.write(json.dumps({"invoice_text": invoice_text, "ground_truth": clean_gt}, ensure_ascii=False) + "\n")
            f.flush()
            seen_hashes.add(h)
            generated += 1
            if generated % 25 == 0:
                print(f"generate_synthetic: {existing + generated}/{n} written", flush=True)


# ---------------------------------------------------------------------------
# §6.3 Chat-format training example
# ---------------------------------------------------------------------------


def build_training_example(invoice_text: str, ground_truth: dict) -> dict:
    """Formats one example into the instruction-tuning chat format (see §6.3)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": invoice_text},
            {"role": "assistant", "content": json.dumps(ground_truth, ensure_ascii=False)},
        ]
    }


# ---------------------------------------------------------------------------
# §5.4 Train/val/test split
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_and_write(examples: list[dict], out_dir: str, seed: int = 42) -> None:
    """Writes train.jsonl, val.jsonl, test.jsonl per §5.4 split rules.

    Each item in `examples` must carry a "source" key of "synthetic" or "sroie" (added by
    the caller / CLI when combining generate_synthetic's and load_sroie's outputs).

    Split logic (§5.4): reserve 10% of the synthetic examples (full schema) and up to 100
    SROIE examples (partial schema) as held-out test sets, never touched during training.
    (10% rather than a hardcoded 100 so this also does the right thing when the synthetic
    pool has been deliberately shrunk below the PRD's 1000 target — see README/
    GENERATION_LOG deviation notes on synthetic dataset size; at the PRD's full 1000 this
    is exactly the specified 100.) The remaining pool per source is split 90/10 into
    train/val, sources kept separate then concatenated so both are represented in each
    split (§5.3 dedup on vendor+total+line_item_count is applied to the synthetic pool
    before splitting).
    """
    rng = random.Random(seed)

    synthetic = [e for e in examples if e.get("source") == "synthetic"]
    sroie = [e for e in examples if e.get("source") == "sroie"]

    # §5.3 dedup (belt-and-suspenders — generate_synthetic already dedups at write time)
    seen: set[str] = set()
    deduped_synthetic = []
    for ex in synthetic:
        h = _hash_key(ex["ground_truth"])
        if h in seen:
            continue
        seen.add(h)
        deduped_synthetic.append(ex)
    synthetic = deduped_synthetic

    rng.shuffle(synthetic)
    rng.shuffle(sroie)

    n_synthetic_test = round(len(synthetic) * 0.1)
    synthetic_test = synthetic[:n_synthetic_test]
    synthetic_pool = synthetic[n_synthetic_test:]

    n_sroie_test = min(100, len(sroie))
    sroie_test = sroie[:n_sroie_test]
    sroie_pool = sroie[n_sroie_test : n_sroie_test + 500]  # "up to 500 SROIE ... for train"

    def _split_90_10(pool: list[dict]) -> tuple[list[dict], list[dict]]:
        cut = round(len(pool) * 0.9)
        return pool[:cut], pool[cut:]

    synthetic_train, synthetic_val = _split_90_10(synthetic_pool)
    sroie_train, sroie_val = _split_90_10(sroie_pool)

    train_rows = [build_training_example(e["invoice_text"], e["ground_truth"]) for e in synthetic_train + sroie_train]
    val_rows = [build_training_example(e["invoice_text"], e["ground_truth"]) for e in synthetic_val + sroie_val]
    test_synthetic_rows = [{"invoice_text": e["invoice_text"], "ground_truth": e["ground_truth"]} for e in synthetic_test]
    test_sroie_rows = [{"invoice_text": e["invoice_text"], "ground_truth": e["ground_truth"]} for e in sroie_test]

    out = Path(out_dir)
    _write_jsonl(out / "train.jsonl", train_rows)
    _write_jsonl(out / "val.jsonl", val_rows)
    _write_jsonl(out / "test_synthetic.jsonl", test_synthetic_rows)
    _write_jsonl(out / "test_sroie.jsonl", test_sroie_rows)

    print(
        f"split_and_write: train={len(train_rows)} val={len(val_rows)} "
        f"test_synthetic={len(test_synthetic_rows)} test_sroie={len(test_sroie_rows)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed train/val/test JSONL files (PRD §5)")
    parser.add_argument("--raw-dir", default="data/raw/sroie")
    parser.add_argument("--synthetic-out", default="data/synthetic/synthetic_raw.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--n-synthetic", type=int, default=1000)
    parser.add_argument("--generator-model", default="openai/gpt-oss-20b")
    parser.add_argument("--skip-generation", action="store_true", help="Use existing synthetic-out as-is")
    args = parser.parse_args()

    if not args.skip_generation:
        generate_synthetic(args.n_synthetic, args.generator_model, args.synthetic_out)

    synthetic_examples = []
    synth_path = Path(args.synthetic_out)
    if synth_path.exists():
        with synth_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["source"] = "synthetic"
                    synthetic_examples.append(row)

    download_sroie(args.raw_dir)
    sroie_examples = load_sroie(args.raw_dir)

    split_and_write(synthetic_examples + sroie_examples, args.processed_dir)


if __name__ == "__main__":
    main()
