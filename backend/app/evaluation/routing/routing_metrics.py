from typing import Dict, Any, List, Optional, Sequence, Set, Union


def route_accuracy(
    predicted_route: Optional[str],
    expected_routes: Union[str, Sequence[str]],
) -> float:
    """
    Returns 1.0 if predicted_route matches any of the expected/acceptable routes, else 0.0.
    """
    if not predicted_route:
        return 0.0

    if isinstance(expected_routes, str):
        expected_set = {expected_routes.lower().strip()}
    else:
        expected_set = {r.lower().strip() for r in expected_routes}

    return 1.0 if predicted_route.lower().strip() in expected_set else 0.0


def multi_route_precision_recall(
    predicted_routes: Sequence[str],
    expected_routes: Sequence[str],
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 for multi-route selection.
    """
    pred_set = {r.lower().strip() for r in predicted_routes}
    exp_set = {r.lower().strip() for r in expected_routes}

    if not pred_set and not exp_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not exp_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    hits = len(pred_set.intersection(exp_set))
    precision = float(hits) / float(len(pred_set))
    recall = float(hits) / float(len(exp_set))
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_unnecessary_vision(
    is_visual_required: bool,
    is_vision_enabled: bool,
    images_attached_count: int = 0,
) -> Dict[str, Any]:
    """
    Detects whether vision was unnecessarily activated (wasting tokens/latency on pure text/tables):
    - unnecessary_vision: True if question didn't require visual understanding but vision was triggered
    - correct_vision_avoidance: True if question didn't require visual understanding and vision was skipped
    - missed_vision: True if visual was required but vision was NOT triggered
    """
    unnecessary = bool(not is_visual_required and (is_vision_enabled or images_attached_count > 0))
    avoided = bool(not is_visual_required and not is_vision_enabled and images_attached_count == 0)
    missed = bool(is_visual_required and not is_vision_enabled)

    return {
        "unnecessary_vision": unnecessary,
        "correct_vision_avoidance": avoided,
        "missed_vision": missed,
    }


def evaluate_unnecessary_agentic(
    is_agentic_required: bool,
    selected_route: str,
    total_steps: int = 1,
) -> Dict[str, Any]:
    """
    Detects whether agentic execution was unnecessarily dispatched for single-hop queries.
    """
    is_agent_route = (selected_route.lower() == "agentic") or (total_steps > 1)
    unnecessary = bool(not is_agentic_required and is_agent_route)
    correct_avoidance = bool(not is_agentic_required and not is_agent_route)
    missed = bool(is_agentic_required and not is_agent_route)

    return {
        "unnecessary_agentic": unnecessary,
        "correct_agentic_avoidance": correct_avoidance,
        "missed_agentic": missed,
    }


def evaluate_missing_index_detection(
    available_indexes: Sequence[str],
    requested_route: str,
    warnings: Sequence[str],
) -> Dict[str, Any]:
    """
    Checks if the system correctly warned when a required index (e.g. pgvector, graph, clip)
    was absent for the requested or selected route.
    """
    route_index_map = {
        "text_dense": "vector_embeddings",
        "multimodal_visual": "clip_embeddings",
        "graph": "graph_entities",
        "multimodal_table": "table_elements",
    }
    needed_index = route_index_map.get(requested_route)
    is_missing = bool(needed_index and needed_index not in available_indexes)

    has_warning = len(warnings) > 0
    correct_detection = (is_missing and has_warning) or (not is_missing and not has_warning)

    return {
        "is_missing": is_missing,
        "has_warning": has_warning,
        "correct_detection": correct_detection,
    }


def evaluate_document_isolation(
    target_document_ids: Sequence[str],
    retrieved_document_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Verifies that all retrieved documents belong strictly to the permitted document scope.
    Computes cross_document_leakage_rate: fraction of retrieved items from outside target scope.
    """
    target_set = {str(d) for d in target_document_ids} if target_document_ids else set()
    retrieved_list = [str(d) for d in retrieved_document_ids]

    if not target_set:
        # No document restriction configured
        return {
            "is_isolated": True,
            "leakage_count": 0,
            "leakage_rate": 0.0,
            "total_retrieved": len(retrieved_list),
        }

    leakage_count = sum(1 for d in retrieved_list if d not in target_set)
    total = len(retrieved_list)
    leakage_rate = float(leakage_count) / float(total) if total > 0 else 0.0

    return {
        "is_isolated": leakage_count == 0,
        "leakage_count": leakage_count,
        "leakage_rate": round(leakage_rate, 4),
        "total_retrieved": total,
    }


def evaluate_fallback_correctness(
    primary_route_available: bool,
    fallback_route_used: Optional[str],
    approved_fallback_routes: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluates whether an appropriate fallback route was chosen when the primary route was unavailable.
    """
    if primary_route_available:
        return {
            "fallback_needed": False,
            "fallback_correct": True,
            "fallback_route_used": fallback_route_used,
        }

    approved_set = {r.lower().strip() for r in approved_fallback_routes}
    is_approved = bool(fallback_route_used and fallback_route_used.lower().strip() in approved_set)

    return {
        "fallback_needed": True,
        "fallback_correct": is_approved,
        "fallback_route_used": fallback_route_used,
    }


def compute_comprehensive_routing_metrics(
    selected_route: str,
    expected_routes: Sequence[str],
    is_visual_required: bool = False,
    is_vision_enabled: bool = False,
    images_attached_count: int = 0,
    is_agentic_required: bool = False,
    target_document_ids: Optional[Sequence[str]] = None,
    retrieved_document_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Computes full routing quality evaluation for a single query.
    """
    acc = route_accuracy(selected_route, expected_routes)
    vis_eval = evaluate_unnecessary_vision(
        is_visual_required=is_visual_required,
        is_vision_enabled=is_vision_enabled,
        images_attached_count=images_attached_count,
    )
    agent_eval = evaluate_unnecessary_agentic(
        is_agentic_required=is_agentic_required,
        selected_route=selected_route,
    )
    doc_eval = evaluate_document_isolation(
        target_document_ids=target_document_ids or [],
        retrieved_document_ids=retrieved_document_ids or [],
    )

    return {
        "route_accuracy": acc,
        "selected_route": selected_route,
        "expected_routes": list(expected_routes),
        "unnecessary_vision": vis_eval["unnecessary_vision"],
        "correct_vision_avoidance": vis_eval["correct_vision_avoidance"],
        "unnecessary_agentic": agent_eval["unnecessary_agentic"],
        "document_isolation": doc_eval["is_isolated"],
        "cross_document_leakage_rate": doc_eval["leakage_rate"],
    }
