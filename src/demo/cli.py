"""Demo CLI: extract structured fields from one invoice using the fine-tuned model
(PRD §8).

    python -m src.demo.cli --input path/to/invoice.txt
"""
import argparse
import json

from src.eval.run_eval import generate_json, load_model_and_tokenizer, parse_model_json
from src.train.train_lora import DEFAULT_BASE_MODEL

_REQUIRED_KEYS = {"vendor", "invoice_number", "invoice_date", "line_items", "subtotal", "tax", "total"}

_model_cache: dict = {}


def _validate_schema(parsed: dict) -> None:
    missing = _REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Model output is missing required schema keys: {sorted(missing)}")
    if not isinstance(parsed.get("line_items"), list):
        raise ValueError("Model output 'line_items' must be a list")
    for item in parsed["line_items"]:
        if not isinstance(item, dict) or not {"description", "quantity", "unit_price", "line_total"} <= item.keys():
            raise ValueError(f"Malformed line_item entry: {item!r}")


def extract(invoice_text: str, adapter_path: str = "models/invoice-lora-v1", base_model: str = DEFAULT_BASE_MODEL) -> dict:
    """Loads base model + LoRA adapter, runs the prompt from §6.3, parses and returns
    the JSON dict. Raises a clear error (not a silent None) if the model output fails
    to parse as valid JSON against the §5.2 schema."""
    cache_key = (base_model, adapter_path)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = load_model_and_tokenizer(base_model, adapter_path)
    model, tokenizer = _model_cache[cache_key]

    raw_output, _latency = generate_json(model, tokenizer, invoice_text)
    parsed = parse_model_json(raw_output)
    if parsed is None:
        raise ValueError(f"Model output failed to parse as JSON:\n{raw_output}")
    _validate_schema(parsed)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured invoice fields (PRD §8)")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--adapter", default="models/invoice-lora-v1")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        invoice_text = f.read()

    result = extract(invoice_text, adapter_path=args.adapter, base_model=args.base_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
