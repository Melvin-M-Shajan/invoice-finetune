"""§11 smoke test: extract() runs end-to-end and produces valid-schema JSON.
Does not assert correctness, only that the pipeline runs. Skipped until a LoRA
adapter has actually been trained (Milestone 3) — see README for how to run it."""
import os

import pytest

from src.demo.cli import extract

ADAPTER_PATH = "models/invoice-lora-v1"
FIXTURE_INVOICE = """Acme Supply Co.
Invoice #: INV-104521
Date: 2024-03-11

1 x Widget A - standard @ $12.50 = $12.50
3 x Office supplies bundle @ $8.00 = $24.00

Subtotal: $36.50
Tax: $2.92
Total: $39.42
"""


@pytest.mark.skipif(not os.path.isdir(ADAPTER_PATH), reason="LoRA adapter not trained yet (Milestone 3)")
def test_extract_smoke():
    result = extract(FIXTURE_INVOICE, adapter_path=ADAPTER_PATH)
    assert isinstance(result, dict)
    assert isinstance(result["vendor"], str)
    assert isinstance(result["line_items"], list)
    assert isinstance(result["total"], (int, float))
