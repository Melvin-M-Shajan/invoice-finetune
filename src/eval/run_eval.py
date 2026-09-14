"""Zero-shot vs. fine-tuned evaluation (PRD §7.2 / §7.3).

    python -m src.eval.run_eval --model <base_model_id> --mode zero-shot \
        --test data/processed/test_synthetic.jsonl --out results/baseline_synthetic.json

    python -m src.eval.run_eval --model <base_model_id> --mode finetuned \
        --adapter models/invoice-lora-v1 --test data/processed/test_synthetic.jsonl \
        --out results/finetuned_synthetic.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.prepare_dataset import SYSTEM_PROMPT
from src.eval.metrics import aggregate, score_example

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_model_and_tokenizer(base_model: str, adapter_path: str | None = None):
    """Loads the base model (4-bit if VRAM < 24GB, per §6.2) and optionally applies a
    LoRA adapter on top. adapter_path=None reproduces the zero-shot baseline exactly."""
    use_4bit = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 24 * (1024**3)
    quant_kwargs = {}
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        quant_kwargs = {"quantization_config": bnb_config}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        **quant_kwargs,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def parse_model_json(raw_output: str) -> dict | None:
    """Extracts and parses the model's JSON object from raw generated text. Returns
    None (not a silent default dict) if it fails to parse — callers must score that as
    0 for every field per §5.2, never drop it from the eval set."""
    if not raw_output:
        return None
    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    match = _JSON_BLOCK_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


@torch.inference_mode()
def generate_json(model, tokenizer, invoice_text: str, max_new_tokens: int = 768) -> tuple[str, float]:
    """Runs one extraction, returns (raw_text_output, latency_seconds)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": invoice_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency = time.perf_counter() - start

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return raw_output, latency


def run_eval(model_id: str, mode: str, test_path: str, out_path: str, adapter_path: str | None = None) -> dict:
    if mode not in ("zero-shot", "finetuned"):
        raise ValueError("mode must be 'zero-shot' or 'finetuned'")
    if mode == "finetuned" and not adapter_path:
        raise ValueError("--adapter is required for --mode finetuned")

    model, tokenizer = load_model_and_tokenizer(model_id, adapter_path if mode == "finetuned" else None)

    examples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    results = []
    all_scores = []
    latencies = []
    for i, ex in enumerate(examples):
        raw_output, latency = generate_json(model, tokenizer, ex["invoice_text"])
        prediction = parse_model_json(raw_output)
        scores = score_example(prediction, ex["ground_truth"])
        all_scores.append(scores)
        latencies.append(latency)
        results.append(
            {
                "invoice_text": ex["invoice_text"],
                "ground_truth": ex["ground_truth"],
                "prediction": prediction,
                "raw_output": raw_output,
                "scores": scores,
                "latency_seconds": latency,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"run_eval [{mode}]: {i + 1}/{len(examples)}")

    report = {
        "model": model_id,
        "mode": mode,
        "adapter": adapter_path if mode == "finetuned" else None,
        "test_path": test_path,
        "n_examples": len(examples),
        "aggregate": aggregate(all_scores),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "examples": results,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_eval: wrote {out_path} (n={len(examples)}, overall_f1={report['aggregate'].get('overall_f1'):.4f})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot vs. fine-tuned evaluation (PRD §7)")
    parser.add_argument("--model", required=True, dest="model_id")
    parser.add_argument("--mode", required=True, choices=["zero-shot", "finetuned"])
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--test", required=True, dest="test_path")
    parser.add_argument("--out", required=True, dest="out_path")
    args = parser.parse_args()

    run_eval(args.model_id, args.mode, args.test_path, args.out_path, adapter_path=args.adapter)


if __name__ == "__main__":
    main()
