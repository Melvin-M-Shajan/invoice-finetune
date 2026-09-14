"""LoRA fine-tuning entry point (PRD §6.4).

Entry point: python -m src.train.train_lora --train data/processed/train.jsonl \
    --val data/processed/val.jsonl --output-dir models/invoice-lora-v1
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_CONFIG_KWARGS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


class JsonlLossLogger(TrainerCallback):
    """Logs train/val loss every `logging_steps` to a local JSONL file (§6.4)."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, **logs}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


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
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train")

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

    lora_config = LoraConfig(**LORA_CONFIG_KWARGS)

    wandb_key = os.environ.get("WANDB_API_KEY")
    report_to = ["wandb"] if wandb_key else "none"

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_seq_length,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        seed=seed,
        report_to=report_to,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[JsonlLossLogger(os.path.join(output_dir, "training_log.jsonl"))],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune the invoice extraction model (PRD §6.4)")
    parser.add_argument("--train", dest="train_path", default="data/processed/train.jsonl")
    parser.add_argument("--val", dest="val_path", default="data/processed/val.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="models/invoice-lora-v1")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
