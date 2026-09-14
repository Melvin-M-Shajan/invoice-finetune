from src.eval.metrics import aggregate, exact_match, normalized_match, score_example


def test_exact_match_string_case_and_whitespace():
    assert exact_match("Acme  Corp", "acme corp") is True
    assert exact_match(" Acme Corp ", "Acme Corp") is True
    assert exact_match("Acme Corp", "Other Corp") is False


def test_exact_match_numbers():
    assert exact_match(100, "100.00") is True
    assert exact_match("$1,200.50", 1200.50) is True
    assert exact_match(100, 100.01) is False


def test_exact_match_none_handling():
    assert exact_match(None, None) is True
    assert exact_match(None, "x") is False
    assert exact_match("x", None) is False


def test_normalized_match_date_formats():
    assert normalized_match("2024-01-05", "01/05/2024", "date") is True
    assert normalized_match("Jan 5, 2024", "2024-01-05", "date") is True
    assert normalized_match("2024-01-05", "2024-02-05", "date") is False


def test_normalized_match_date_falls_back_to_exact_match_on_parse_failure():
    assert normalized_match("not-a-date", "not-a-date", "date") is True
    assert normalized_match("not-a-date", "also-not-a-date", "date") is False


def test_normalized_match_number_tolerance():
    assert normalized_match(100.004, 100.0, "number") is True
    assert normalized_match(100.02, 100.0, "number") is False
    assert normalized_match("$100.00", "100", "number") is True


def test_score_example_full_schema():
    true_json = {
        "vendor": "Acme Corp",
        "invoice_number": "INV-1",
        "invoice_date": "2024-01-05",
        "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 5.0, "line_total": 10.0}],
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
    }
    pred_json = {
        "vendor": "acme corp",
        "invoice_number": "INV-2",
        "invoice_date": "01/05/2024",
        "line_items": [{"description": "widget", "quantity": 2, "unit_price": 5.0, "line_total": 10.0}],
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
    }
    scores = score_example(pred_json, true_json)
    assert scores["vendor"] is True
    assert scores["invoice_number"] is False
    assert scores["invoice_date"] is True
    assert scores["subtotal"] is True
    assert scores["line_items_precision"] == 1.0
    assert scores["line_items_recall"] == 1.0


def test_score_example_partial_schema_sroie():
    true_json = {"vendor": "Acme Corp", "invoice_date": "2024-01-05", "total": 11.0}
    pred_json = {"vendor": "Acme Corp", "invoice_date": "2024-01-05", "total": 999.0}
    scores = score_example(pred_json, true_json)
    assert set(scores.keys()) == {"vendor", "invoice_date", "total"}
    assert scores["total"] is False


def test_score_example_malformed_json_is_all_false():
    true_json = {"vendor": "Acme Corp", "total": 11.0, "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 1.0, "line_total": 1.0}]}
    scores = score_example(None, true_json)
    assert scores["vendor"] is False
    assert scores["total"] is False
    assert scores["line_items_precision"] == 0.0
    assert scores["line_items_recall"] == 0.0


def test_score_example_line_items_partial_overlap():
    true_json = {
        "line_items": [
            {"description": "Widget A", "quantity": 1, "unit_price": 5.0, "line_total": 5.0},
            {"description": "Widget B", "quantity": 2, "unit_price": 3.0, "line_total": 6.0},
            {"description": "Widget C", "quantity": 1, "unit_price": 7.0, "line_total": 7.0},
        ]
    }
    pred_json = {
        "line_items": [
            {"description": "Widget A", "quantity": 1, "unit_price": 5.0, "line_total": 5.0},
            {"description": "Widget X", "quantity": 9, "unit_price": 9.0, "line_total": 81.0},
        ]
    }
    scores = score_example(pred_json, true_json)
    assert scores["line_items_precision"] == 0.5
    assert abs(scores["line_items_recall"] - (1 / 3)) < 1e-9


def test_score_example_line_items_duplicate_pred_does_not_double_match():
    true_json = {"line_items": [{"description": "Widget A", "quantity": 1, "unit_price": 5.0, "line_total": 5.0}]}
    pred_json = {
        "line_items": [
            {"description": "Widget A", "quantity": 1, "unit_price": 5.0, "line_total": 5.0},
            {"description": "Widget A", "quantity": 1, "unit_price": 5.0, "line_total": 5.0},
        ]
    }
    scores = score_example(pred_json, true_json)
    assert scores["line_items_precision"] == 0.5
    assert scores["line_items_recall"] == 1.0


def test_aggregate_basic():
    all_scores = [
        {"vendor": True, "total": True, "line_items_precision": 1.0, "line_items_recall": 1.0},
        {"vendor": False, "total": True, "line_items_precision": 0.0, "line_items_recall": 0.0},
    ]
    result = aggregate(all_scores)
    assert result["vendor"] == 0.5
    assert result["total"] == 1.0
    assert result["line_items_precision"] == 0.5
    assert result["line_items_recall"] == 0.5
    assert 0.0 <= result["overall_f1"] <= 1.0


def test_aggregate_handles_partial_schema_examples_mixed_in():
    all_scores = [
        {"vendor": True, "invoice_date": True, "total": True},
        {"vendor": True, "invoice_number": False, "invoice_date": True, "total": False, "line_items_precision": 0.5, "line_items_recall": 0.5},
    ]
    result = aggregate(all_scores)
    assert result["vendor"] == 1.0
    assert result["invoice_number"] == 0.0
    assert result["total"] == 0.5
