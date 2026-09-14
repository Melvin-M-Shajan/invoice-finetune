"""Field-level scoring for invoice extraction (PRD §7.1)."""
import re
from datetime import datetime

from dateutil import parser as dateutil_parser

SCALAR_FIELD_TYPES = {
    "vendor": "string",
    "invoice_number": "string",
    "invoice_date": "date",
    "subtotal": "number",
    "tax": "number",
    "total": "number",
}

_CURRENCY_CHARS = re.compile(r"[,$€£₹]")
_WHITESPACE = re.compile(r"\s+")


def _normalize_str(value) -> str:
    return _WHITESPACE.sub(" ", str(value).strip().lower())


def _try_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = _CURRENCY_CHARS.sub("", value).strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _try_parse_date(value):
    if isinstance(value, str):
        try:
            dt = dateutil_parser.parse(value, fuzzy=True)
            return dt.date().isoformat()
        except (ValueError, OverflowError):
            return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    return None


def exact_match(pred_value, true_value) -> bool:
    """Case-insensitive, whitespace-normalized string equality. For numbers, compare
    after parsing to float."""
    if pred_value is None and true_value is None:
        return True
    if pred_value is None or true_value is None:
        return False
    pred_num, true_num = _try_float(pred_value), _try_float(true_value)
    if pred_num is not None and true_num is not None:
        return pred_num == true_num
    return _normalize_str(pred_value) == _normalize_str(true_value)


def normalized_match(pred_value, true_value, field_type: str) -> bool:
    """field_type in {"string", "date", "number"}. Dates: parse both to ISO
    YYYY-MM-DD before comparing (accept common formats). Numbers: match if
    abs(pred - true) < 0.01. Falls back to exact_match for anything that fails to
    parse."""
    if pred_value is None and true_value is None:
        return True
    if pred_value is None or true_value is None:
        return False

    if field_type == "date":
        pred_date, true_date = _try_parse_date(pred_value), _try_parse_date(true_value)
        if pred_date is not None and true_date is not None:
            return pred_date == true_date
        return exact_match(pred_value, true_value)

    if field_type == "number":
        pred_num, true_num = _try_float(pred_value), _try_float(true_value)
        if pred_num is not None and true_num is not None:
            return abs(pred_num - true_num) < 0.01
        return exact_match(pred_value, true_value)

    # "string"
    return exact_match(pred_value, true_value)


def _score_line_items(pred_items: list, true_items: list) -> tuple[float, float]:
    """Set-based matching: a predicted line item is a TP if there is an unmatched
    true line item with normalized_match True on description AND quantity AND
    unit_price."""
    matched_true_idx: set[int] = set()
    true_positives = 0
    for pred_item in pred_items:
        for idx, true_item in enumerate(true_items):
            if idx in matched_true_idx:
                continue
            if (
                normalized_match(pred_item.get("description"), true_item.get("description"), "string")
                and normalized_match(pred_item.get("quantity"), true_item.get("quantity"), "number")
                and normalized_match(pred_item.get("unit_price"), true_item.get("unit_price"), "number")
            ):
                matched_true_idx.add(idx)
                true_positives += 1
                break

    if pred_items:
        precision = true_positives / len(pred_items)
    else:
        precision = 1.0 if not true_items else 0.0

    if true_items:
        recall = true_positives / len(true_items)
    else:
        recall = 1.0 if not pred_items else 0.0

    return precision, recall


def score_example(pred_json: dict | None, true_json: dict) -> dict:
    """Returns per-field bool correctness, e.g. {"vendor": True, "invoice_number":
    False, ...}. If pred_json is None (failed to parse as valid JSON), every field
    is False. Line items are scored as a set: a predicted line item counts correct
    if there exists an unmatched true line item with normalized_match True on
    description AND quantity AND unit_price; report line_items_precision and
    line_items_recall separately from the scalar fields.

    Only fields present as keys in true_json are scored (supports SROIE's partial
    schema: vendor/invoice_date/total only)."""
    scores: dict = {}

    for field, field_type in SCALAR_FIELD_TYPES.items():
        if field not in true_json:
            continue
        true_value = true_json[field]
        if pred_json is None:
            scores[field] = False
            continue
        pred_value = pred_json.get(field)
        scores[field] = normalized_match(pred_value, true_value, field_type)

    if "line_items" in true_json:
        true_items = true_json.get("line_items") or []
        pred_items = (pred_json.get("line_items") if pred_json is not None else None) or []
        precision, recall = _score_line_items(pred_items, true_items)
        scores["line_items_precision"] = precision
        scores["line_items_recall"] = recall

    return scores


def aggregate(all_scores: list[dict]) -> dict:
    """Returns per-field accuracy (mean of bool correctness across examples), plus
    overall F1 computed as 2PR/(P+R) where P = mean per-field precision, R = mean
    per-field recall, treating each field as a binary classification
    (present+correct vs. not).

    Scalar fields have no separate "field absent from prediction" signal beyond the
    bool itself, so for them precision == recall == accuracy (every example is a
    positive instance since the schema always defines the field). line_items
    contributes its own real precision/recall (averaged across examples) to the P
    and R means instead of a single accuracy number.
    """
    scalar_fields = [f for f in SCALAR_FIELD_TYPES if any(f in s for s in all_scores)]
    result: dict = {}

    for field in scalar_fields:
        values = [s[field] for s in all_scores if field in s]
        result[field] = sum(values) / len(values) if values else 0.0

    has_line_items = any("line_items_precision" in s for s in all_scores)
    if has_line_items:
        precisions = [s["line_items_precision"] for s in all_scores if "line_items_precision" in s]
        recalls = [s["line_items_recall"] for s in all_scores if "line_items_recall" in s]
        result["line_items_precision"] = sum(precisions) / len(precisions) if precisions else 0.0
        result["line_items_recall"] = sum(recalls) / len(recalls) if recalls else 0.0

    p_terms = [result[f] for f in scalar_fields]
    r_terms = [result[f] for f in scalar_fields]
    if has_line_items:
        p_terms.append(result["line_items_precision"])
        r_terms.append(result["line_items_recall"])

    p_mean = sum(p_terms) / len(p_terms) if p_terms else 0.0
    r_mean = sum(r_terms) / len(r_terms) if r_terms else 0.0
    result["overall_f1"] = (2 * p_mean * r_mean / (p_mean + r_mean)) if (p_mean + r_mean) > 0 else 0.0

    return result
