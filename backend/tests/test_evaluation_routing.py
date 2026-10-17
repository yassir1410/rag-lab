import pytest
from app.evaluation.routing.routing_metrics import (
    route_accuracy,
    multi_route_precision_recall,
    evaluate_unnecessary_vision,
    evaluate_unnecessary_agentic,
    evaluate_missing_index_detection,
    evaluate_document_isolation,
    evaluate_fallback_correctness,
    compute_comprehensive_routing_metrics,
)


def test_route_accuracy():
    # Exact single match
    assert route_accuracy("multimodal_table", "multimodal_table") == 1.0
    # Case insensitivity
    assert route_accuracy("Multimodal_Table", "multimodal_table") == 1.0
    # Match in list of acceptable routes
    assert route_accuracy("text_dense", ["text_dense", "text_hybrid"]) == 1.0
    # Mismatch
    assert route_accuracy("agentic", ["text_dense", "text_hybrid"]) == 0.0
    # None predicted
    assert route_accuracy(None, ["text_dense"]) == 0.0


def test_multi_route_precision_recall():
    pred = ["text_dense", "text_hybrid"]
    expected = ["text_hybrid", "multimodal_table"]
    # Shared: "text_hybrid" (1 hit)
    # Precision: 1 / 2 = 0.5
    # Recall: 1 / 2 = 0.5
    # F1: 0.5
    res = multi_route_precision_recall(pred, expected)
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5


def test_evaluate_unnecessary_vision():
    # Case 1: Pure tabular query, vision was avoided
    res = evaluate_unnecessary_vision(
        is_visual_required=False,
        is_vision_enabled=False,
        images_attached_count=0,
    )
    assert res["unnecessary_vision"] is False
    assert res["correct_vision_avoidance"] is True
    assert res["missed_vision"] is False

    # Case 2: Pure tabular query, but vision was unnecessarily triggered
    res = evaluate_unnecessary_vision(
        is_visual_required=False,
        is_vision_enabled=True,
        images_attached_count=1,
    )
    assert res["unnecessary_vision"] is True
    assert res["correct_vision_avoidance"] is False

    # Case 3: Genuine visual query, vision was correctly enabled
    res = evaluate_unnecessary_vision(
        is_visual_required=True,
        is_vision_enabled=True,
        images_attached_count=1,
    )
    assert res["unnecessary_vision"] is False
    assert res["missed_vision"] is False


def test_evaluate_unnecessary_agentic():
    # Simple query routed to dense search
    res = evaluate_unnecessary_agentic(
        is_agentic_required=False,
        selected_route="text_dense",
        total_steps=1,
    )
    assert res["unnecessary_agentic"] is False
    assert res["correct_agentic_avoidance"] is True

    # Simple query triggered an agentic multi-step loop
    res = evaluate_unnecessary_agentic(
        is_agentic_required=False,
        selected_route="agentic",
        total_steps=3,
    )
    assert res["unnecessary_agentic"] is True
    assert res["correct_agentic_avoidance"] is False


def test_evaluate_missing_index_detection():
    # Required vector index missing, system provided warning
    res = evaluate_missing_index_detection(
        available_indexes=[],
        requested_route="text_dense",
        warnings=["Vector embeddings are missing for this document."],
    )
    assert res["is_missing"] is True
    assert res["correct_detection"] is True

    # Required vector index present, no warning
    res = evaluate_missing_index_detection(
        available_indexes=["vector_embeddings"],
        requested_route="text_dense",
        warnings=[],
    )
    assert res["is_missing"] is False
    assert res["correct_detection"] is True


def test_evaluate_document_isolation():
    target_docs = ["doc-1"]
    retrieved_docs = ["doc-1", "doc-1", "doc-1"]
    res = evaluate_document_isolation(target_docs, retrieved_docs)
    assert res["is_isolated"] is True
    assert res["leakage_count"] == 0
    assert res["leakage_rate"] == 0.0

    # Cross-document leakage
    retrieved_with_leakage = ["doc-1", "doc-2", "doc-1"]
    res = evaluate_document_isolation(target_docs, retrieved_with_leakage)
    assert res["is_isolated"] is False
    assert res["leakage_count"] == 1
    assert res["leakage_rate"] == round(1.0 / 3.0, 4)


def test_evaluate_fallback_correctness():
    # Primary available
    res = evaluate_fallback_correctness(
        primary_route_available=True,
        fallback_route_used=None,
        approved_fallback_routes=["multimodal_table"],
    )
    assert res["fallback_needed"] is False
    assert res["fallback_correct"] is True

    # Primary unavailable, picked approved fallback
    res = evaluate_fallback_correctness(
        primary_route_available=False,
        fallback_route_used="multimodal_table",
        approved_fallback_routes=["multimodal_table", "text_hybrid"],
    )
    assert res["fallback_needed"] is True
    assert res["fallback_correct"] is True

    # Primary unavailable, picked unapproved fallback
    res = evaluate_fallback_correctness(
        primary_route_available=False,
        fallback_route_used="random_route",
        approved_fallback_routes=["multimodal_table"],
    )
    assert res["fallback_correct"] is False


def test_compute_comprehensive_routing_metrics():
    res = compute_comprehensive_routing_metrics(
        selected_route="multimodal_table",
        expected_routes=["multimodal_table"],
        is_visual_required=False,
        is_vision_enabled=False,
        images_attached_count=0,
        is_agentic_required=False,
        target_document_ids=["doc-solar-system"],
        retrieved_document_ids=["doc-solar-system"],
    )
    assert res["route_accuracy"] == 1.0
    assert res["unnecessary_vision"] is False
    assert res["correct_vision_avoidance"] is True
    assert res["document_isolation"] is True
    assert res["cross_document_leakage_rate"] == 0.0
