import pytest
from app.evaluation.generation.accuracy import (
    exact_match,
    token_f1,
    normalized_answer_match,
    extract_numbers,
    numerical_correctness,
    compute_generation_accuracy,
)


def test_exact_match_basic():
    # Identical strings
    assert exact_match("Jupiter", "Jupiter") == 1.0
    # Case insensitivity
    assert exact_match("jupiter", "Jupiter") == 1.0
    # Whitespace normalization
    assert exact_match("  Jupiter  ", "Jupiter") == 1.0
    # Punctuation stripping
    assert exact_match("Jupiter.", "Jupiter") == 1.0
    # Mismatch
    assert exact_match("Saturn", "Jupiter") == 0.0
    # Both empty
    assert exact_match("", "") == 1.0
    # One empty
    assert exact_match("Jupiter", "") == 0.0


def test_token_f1_calculation():
    # Identical
    res = token_f1("Jupiter is a gas giant", "Jupiter is a gas giant")
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0

    # Partial overlap:
    # Pred: ["jupiter", "is", "large"] (3 tokens)
    # Ref: ["jupiter", "is", "a", "gas", "giant"] (5 tokens)
    # Overlap: "jupiter", "is" (2 tokens)
    # Precision = 2/3, Recall = 2/5 = 0.4
    # F1 = 2 * (2/3 * 0.4) / (2/3 + 0.4) = 2 * (0.2667) / (1.0667) = 0.5
    res = token_f1("Jupiter is large", "Jupiter is a gas giant")
    assert res["precision"] == round(2.0 / 3.0, 4)
    assert res["recall"] == 0.4
    assert res["f1"] == 0.5

    # Zero overlap
    res = token_f1("Mars", "Jupiter")
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_normalized_answer_match():
    acceptable = [
        "23.1 m/s² [TABLE_P1]",
        "Jupiter's surface gravity is 23.1 m/s².",
        "23.1 m/s²",
    ]
    # Exact normalized match with one of the candidates
    assert normalized_answer_match("23.1 m/s²", acceptable) == 1.0
    # Token F1 match above threshold
    assert normalized_answer_match("The surface gravity of Jupiter is 23.1 m/s².", acceptable) == 1.0
    # Completely unrelated answer
    assert normalized_answer_match("Saturn has rings.", acceptable) == 0.0


def test_extract_numbers():
    assert extract_numbers("The speed is 23.1 m/s and mass is -5.2 kg") == [23.1, -5.2]
    assert extract_numbers("Scientific: 1.5e-3 and integer 42") == [0.0015, 42.0]
    assert extract_numbers("No numbers here") == []


def test_numerical_correctness():
    # Correct value within tolerance with matching unit
    res = numerical_correctness(
        predicted="According to page 1, Jupiter's gravity is 23.1 m/s².",
        expected_value=23.1,
        tolerance=0.5,
        expected_unit="m/s²",
    )
    assert res["is_correct"] is True
    assert res["matched_value"] == 23.1
    assert res["unit_matched"] is True
    assert res["error"] == 0.0

    # Value outside tolerance
    res = numerical_correctness(
        predicted="Jupiter's gravity is 28.5 m/s².",
        expected_value=23.1,
        tolerance=0.5,
        expected_unit="m/s²",
    )
    assert res["is_correct"] is False

    # Unit mismatch
    res = numerical_correctness(
        predicted="Jupiter's gravity is 23.1 km/h.",
        expected_value=23.1,
        tolerance=0.5,
        expected_unit="m/s²",
    )
    assert res["is_correct"] is False
    assert res["unit_matched"] is False


def test_compute_generation_accuracy_aggregate():
    pred = "Jupiter's surface gravity is 23.1 m/s²."
    ref = "23.1 m/s²"
    num_spec = {"value": 23.1, "tolerance": 0.5, "unit": "m/s²"}

    res = compute_generation_accuracy(
        predicted=pred,
        reference=ref,
        acceptable_answers=["23.1 m/s²", "23.1"],
        numerical_spec=num_spec,
    )

    assert "exact_match" in res
    assert "token_f1" in res
    assert "normalized_answer_match" in res
    assert "numerical_correctness" in res
    assert res["normalized_answer_match"] == 1.0
    assert res["numerical_correctness"]["is_correct"] is True
