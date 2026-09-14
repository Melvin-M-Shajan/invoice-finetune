import json
from pathlib import Path

from src.data.prepare_dataset import (
    SYSTEM_PROMPT,
    _hash_key,
    _make_ground_truth,
    build_training_example,
    load_sroie,
    split_and_write,
)
import random


def test_build_training_example_matches_prd_format():
    ground_truth = {"vendor": "Acme Corp", "total": 11.0}
    example = build_training_example("raw invoice text here", ground_truth)

    assert list(example.keys()) == ["messages"]
    messages = example["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "raw invoice text here"}
    assert messages[2]["role"] == "assistant"
    assert json.loads(messages[2]["content"]) == ground_truth


def test_make_ground_truth_totals_are_internally_consistent():
    rng = random.Random(1)
    for _ in range(20):
        gt = _make_ground_truth(rng)
        computed_subtotal = round(sum(li["line_total"] for li in gt["line_items"]), 2)
        expected_total = round(computed_subtotal + (gt["tax"] or 0.0), 2)
        assert abs(gt["total"] - expected_total) < 0.01
        assert 1 <= len(gt["line_items"]) <= 6


def test_hash_key_dedup_on_vendor_total_line_count():
    gt_a = {"vendor": "Acme", "total": 10.0, "line_items": [{"description": "x"}]}
    gt_b = {"vendor": "Acme", "total": 10.0, "line_items": [{"description": "y"}]}  # same len
    gt_c = {"vendor": "Acme", "total": 20.0, "line_items": [{"description": "x"}]}
    assert _hash_key(gt_a) == _hash_key(gt_b)
    assert _hash_key(gt_a) != _hash_key(gt_c)


def test_load_sroie_from_fixture(tmp_path: Path):
    doc_dir = tmp_path
    (doc_dir / "X001.txt").write_text(
        "10,10,100,10,100,30,10,30,ACME CORP\n20,40,100,40,100,60,20,60,TOTAL: 25.50\n",
        encoding="utf-8",
    )
    (doc_dir / "X001.json").write_text(
        json.dumps({"company": "ACME CORP", "date": "20/03/2018", "address": "1 Main St", "total": "25.50"}),
        encoding="utf-8",
    )

    examples = load_sroie(str(tmp_path))
    assert len(examples) == 1
    ex = examples[0]
    assert ex["source"] == "sroie"
    assert ex["ground_truth"]["vendor"] == "ACME CORP"
    assert ex["ground_truth"]["invoice_date"] == "2018-03-20"
    assert ex["ground_truth"]["total"] == 25.50
    assert "address" not in ex["ground_truth"]  # §5.1: SROIE address is not part of target schema
    assert "ACME CORP" in ex["invoice_text"]


def test_load_sroie_missing_dir_returns_empty_list(tmp_path: Path):
    assert load_sroie(str(tmp_path / "does_not_exist")) == []


def test_split_and_write_sizes_and_held_out_test(tmp_path: Path):
    rng = random.Random(0)
    synthetic_examples = []
    for i in range(150):
        gt = _make_ground_truth(rng)
        gt["vendor"] = f"Vendor {i}"  # force unique hash per example
        synthetic_examples.append({"invoice_text": f"synthetic invoice {i}", "ground_truth": gt, "source": "synthetic"})

    sroie_examples = [
        {"invoice_text": f"sroie invoice {i}", "ground_truth": {"vendor": f"S{i}", "invoice_date": "2020-01-01", "total": float(i)}, "source": "sroie"}
        for i in range(60)
    ]

    out_dir = tmp_path / "processed"
    split_and_write(synthetic_examples + sroie_examples, str(out_dir), seed=42)

    train = [json.loads(l) for l in (out_dir / "train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (out_dir / "val.jsonl").read_text().splitlines()]
    test_synthetic = [json.loads(l) for l in (out_dir / "test_synthetic.jsonl").read_text().splitlines()]
    test_sroie = [json.loads(l) for l in (out_dir / "test_sroie.jsonl").read_text().splitlines()]

    # 150 synthetic -> 10% (15) held out as test, 135 remain -> 90/10 split (round-half-even)
    assert len(test_synthetic) == 15
    # 60 sroie -> min(100,60)=60 held out as test, 0 remain for train/val
    assert len(test_sroie) == 60
    assert len(train) == 122
    assert len(val) == 13

    # train/val entries are chat-formatted; test entries are raw invoice_text+ground_truth
    assert "messages" in train[0]
    assert "messages" not in test_synthetic[0]
    assert "ground_truth" in test_synthetic[0]
