import re
import string
from typing import Dict, Any, List, Optional, Sequence, Union


def _normalize_text(text: Optional[str]) -> str:
    """Lowercases, handles apostrophes, removes punctuation, and standardizes whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Separate contractions and possessives so "Jupiter's" becomes "jupiter s" or "jupiter"
    text = text.replace("'", " ").replace("’", " ")
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    return " ".join(text.split())


def _tokenize(text: Optional[str]) -> List[str]:
    """Tokenizes normalized text into word tokens."""
    norm = _normalize_text(text)
    return norm.split() if norm else []


def exact_match(predicted: Optional[str], reference: Optional[str]) -> float:
    """
    Exact Match (EM) metric: 1.0 if normalized strings are identical, else 0.0.
    """
    norm_pred = _normalize_text(predicted)
    norm_ref = _normalize_text(reference)
    if not norm_pred and not norm_ref:
        return 1.0
    if not norm_pred or not norm_ref:
        return 0.0
    return 1.0 if norm_pred == norm_ref else 0.0


def token_f1(predicted: Optional[str], reference: Optional[str]) -> Dict[str, float]:
    """
    Computes token-level Precision, Recall, and F1 score between predicted and reference text.
    Handles unigram overlap.
    """
    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Count token frequencies in reference
    ref_counts: Dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    # Count shared tokens
    shared_count = 0
    pred_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
        if ref_counts.get(tok, 0) >= pred_counts[tok]:
            shared_count += 1

    precision = float(shared_count) / float(len(pred_tokens))
    recall = float(shared_count) / float(len(ref_tokens))

    if (precision + recall) <= 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def normalized_answer_match(
    predicted: Optional[str],
    acceptable_answers: Sequence[str],
    f1_threshold: float = 0.70,
) -> float:
    """
    Checks if predicted answer matches any acceptable answer:
    - 1.0 if exact match with any acceptable answer
    - 1.0 if candidate answer is a normalized substring of predicted text
    - 1.0 if token F1 >= f1_threshold with any acceptable answer
    - 1.0 if candidate token recall == 1.0 (all key candidate words present in generated answer)
    - 0.0 otherwise
    """
    if not acceptable_answers:
        return 0.0

    norm_pred = _normalize_text(predicted)
    if not norm_pred:
        return 0.0

    for cand in acceptable_answers:
        norm_cand = _normalize_text(cand)
        if not norm_cand:
            continue
        if norm_pred == norm_cand or norm_cand in norm_pred:
            return 1.0
        scores = token_f1(predicted, cand)
        if scores["f1"] >= f1_threshold or scores["recall"] >= 0.99:
            return 1.0

    return 0.0


def extract_numbers(text: Optional[str]) -> List[float]:
    """
    Extracts floating-point and integer numbers from text.
    Handles negative numbers, decimal points, and scientific notation (e.g., 23.1, -5, 1.2e-3).
    """
    if not text:
        return []
    pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            val = float(m)
            numbers.append(val)
        except ValueError:
            continue
    return numbers


def numerical_correctness(
    predicted: Optional[str],
    expected_value: float,
    tolerance: float = 0.01,
    expected_unit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether predicted text contains a numerical value within tolerance of expected_value.
    Optionally verifies that expected_unit (e.g., 'm/s²', 'kg', 'km', '%') is present in text.
    """
    if not predicted:
        return {
            "is_correct": False,
            "matched_value": None,
            "unit_matched": False,
            "error": None,
        }

    extracted = extract_numbers(predicted)
    if not extracted:
        return {
            "is_correct": False,
            "matched_value": None,
            "unit_matched": False,
            "error": None,
        }

    # Find the extracted number closest to expected_value
    best_diff = float("inf")
    best_val = None
    for val in extracted:
        diff = abs(val - expected_value)
        if diff < best_diff:
            best_diff = diff
            best_val = val

    is_within_tolerance = best_diff <= tolerance

    # Unit matching
    unit_matched = True
    if expected_unit:
        # Check if normalized unit substring is in normalized predicted text
        norm_pred = predicted.lower().replace(" ", "")
        norm_unit = expected_unit.lower().replace(" ", "")
        unit_matched = norm_unit in norm_pred

    is_correct = bool(is_within_tolerance and unit_matched)

    return {
        "is_correct": is_correct,
        "matched_value": best_val,
        "unit_matched": unit_matched,
        "error": round(best_diff, 4) if best_val is not None else None,
    }


def compute_generation_accuracy(
    predicted: Optional[str],
    reference: Optional[str],
    acceptable_answers: Optional[Sequence[str]] = None,
    numerical_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Aggregates deterministic generation accuracy metrics:
    - exact_match: 0.0 or 1.0
    - token_f1: precision, recall, f1
    - normalized_answer_match: 0.0 or 1.0
    - numerical_correctness: details if numerical_spec provided
    """
    em = exact_match(predicted, reference)
    tf1 = token_f1(predicted, reference)

    cand_list = list(acceptable_answers or [])
    if reference and reference not in cand_list:
        cand_list.append(reference)

    norm_match = normalized_answer_match(predicted, cand_list)

    result: Dict[str, Any] = {
        "exact_match": em,
        "token_precision": tf1["precision"],
        "token_recall": tf1["recall"],
        "token_f1": tf1["f1"],
        "normalized_answer_match": norm_match,
    }

    if numerical_spec and isinstance(numerical_spec, dict) and "value" in numerical_spec:
        exp_val = float(numerical_spec["value"])
        tol = float(numerical_spec.get("tolerance", 0.01))
        unit = numerical_spec.get("unit")
        num_res = numerical_correctness(predicted, exp_val, tolerance=tol, expected_unit=unit)
        result["numerical_correctness"] = num_res

    return result
