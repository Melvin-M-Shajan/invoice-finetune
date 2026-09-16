"""Optional Gradio demo UI (PRD §8): textbox in -> JSON out.

    python -m src.demo.app
"""
import json

import gradio as gr

from src.demo.cli import extract

_EXAMPLE_INVOICE = """Northwind Logistics
Invoice #: INV-778812
Date: 2025-06-14

Description                  Qty   Unit Price   Line Total
Freight and handling          3      $145.00       $435.00
Consulting services (hourly)  6      $210.50      $1263.00

Subtotal: $1698.00
Tax: $135.84
Total: $1833.84
"""


def _run_extraction(invoice_text: str) -> dict:
    if not invoice_text or not invoice_text.strip():
        return {"error": "Paste some invoice text first."}
    try:
        return extract(invoice_text)
    except Exception as exc:  # noqa: BLE001 - UI boundary, surface any failure as JSON
        return {"error": str(exc)}


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Invoice Extraction Demo") as demo:
        gr.Markdown(
            "## Invoice Extraction Demo\n"
            "Paste raw invoice text below. Runs the LoRA-fine-tuned "
            "`Qwen2.5-1.5B-Instruct` model (`models/invoice-lora-v1`) and returns the "
            "extracted fields as JSON (PRD §5.2 schema)."
        )
        with gr.Row():
            input_box = gr.Textbox(
                label="Invoice text",
                lines=16,
                value=_EXAMPLE_INVOICE,
                placeholder="Paste raw invoice text here...",
            )
            output_box = gr.JSON(label="Extracted fields")
        run_button = gr.Button("Extract", variant="primary")
        run_button.click(fn=_run_extraction, inputs=input_box, outputs=output_box)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
