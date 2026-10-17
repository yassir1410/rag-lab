import math
from typing import Sequence, Set, Dict, Any, Optional, Union, List


def _deduplicate_preserve_order(items: Sequence[Any]) -> List[str]:
    """Preserves ranking order while filtering out duplicate chunk/document IDs."""
    seen = set()
    result = []
    for item in items:
        str_id = str(item)
        if str_id not in seen:
            seen.add(str_id)
            result.append(str_id)
    return result


def precision_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any]],
    k: int = 5,
) -> float:
    """
    Precision@K = (Number of relevant items in top-K) / K

    Fraction of retrieved chunks among the first K results that are relevant.
    """
    if k <= 0:
        return 0.0

    retrieved = _deduplicate_preserve_order(retrieved_ids)[:k]
    relevant_set = {str(r) for r in relevant_ids}

    if not retrieved or not relevant_set:
        return 0.0

    relevant_retrieved = sum(1 for item in retrieved if item in relevant_set)
    return float(relevant_retrieved) / float(k)


def recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any]],
    k: int = 5,
    no_relevant_policy: float = 0.0,
) -> float:
    """
    Recall@K = (Number of relevant items in top-K) / (Total number of relevant items)

    Fraction of all relevant chunks that were successfully retrieved in the first K results.
    If no relevant items are defined for the question, returns `no_relevant_policy`.
    """
    if k <= 0:
        return 0.0

    relevant_set = {str(r) for r in relevant_ids}
    if not relevant_set:
        return no_relevant_policy

    retrieved = _deduplicate_preserve_order(retrieved_ids)[:k]
    relevant_retrieved = sum(1 for item in retrieved if item in relevant_set)

    return float(relevant_retrieved) / float(len(relevant_set))


def f1_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any]],
    k: int = 5,
) -> float:
    """
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Harmonic mean of Precision@K and Recall@K.
    Returns 0.0 if Precision@K + Recall@K == 0.0.
    """
    if k <= 0:
        return 0.0
    p_k = precision_at_k(retrieved_ids, relevant_ids, k=k)
    r_k = recall_at_k(retrieved_ids, relevant_ids, k=k)
    if (p_k + r_k) <= 0.0:
        return 0.0
    return float(2.0 * p_k * r_k / (p_k + r_k))


def hit_rate_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any]],
    k: int = 5,
) -> float:
    """
    Hit Rate@K (Success Rate) = 1.0 if at least one relevant item appears in top-K, else 0.0.
    """
    if k <= 0:
        return 0.0

    retrieved = _deduplicate_preserve_order(retrieved_ids)[:k]
    relevant_set = {str(r) for r in relevant_ids}

    if not retrieved or not relevant_set:
        return 0.0

    return 1.0 if any(item in relevant_set for item in retrieved) else 0.0


def reciprocal_rank(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any]],
    k: Optional[int] = None,
) -> float:
    """
    Reciprocal Rank (RR) = 1.0 / rank of the first relevant result.
    If no relevant result appears within K (or entire list if k is None), returns 0.0.
    """
    retrieved = _deduplicate_preserve_order(retrieved_ids)
    if k is not None and k > 0:
        retrieved = retrieved[:k]

    relevant_set = {str(r) for r in relevant_ids}
    if not retrieved or not relevant_set:
        return 0.0

    for rank_idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / float(rank_idx)

    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """
    Mean Reciprocal Rank (MRR) across evaluation queries.
    """
    if not reciprocal_ranks:
        return 0.0
    return float(sum(reciprocal_ranks)) / float(len(reciprocal_ranks))


def average_precision(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any]],
    k: Optional[int] = None,
) -> float:
    """
    Average Precision (AP) = sum_{i=1..K} (Precision@i * is_relevant(i)) / |Relevant|
    """
    relevant_set = {str(r) for r in relevant_ids}
    if not relevant_set:
        return 0.0

    retrieved = _deduplicate_preserve_order(retrieved_ids)
    if k is not None and k > 0:
        retrieved = retrieved[:k]

    if not retrieved:
        return 0.0

    running_relevant_count = 0
    precision_sum = 0.0

    for idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            running_relevant_count += 1
            precision_at_i = running_relevant_count / idx
            precision_sum += precision_at_i

    return float(precision_sum) / float(len(relevant_set))


def mean_average_precision(average_precisions: Sequence[float]) -> float:
    """
    Mean Average Precision (MAP) across multiple evaluation questions.
    """
    if not average_precisions:
        return 0.0
    return float(sum(average_precisions)) / float(len(average_precisions))


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    relevance_scores: Union[Set[Any], Sequence[Any], Dict[Any, float]],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@K).
    Supports binary relevance (via set/list of IDs) or graded relevance (via dict of {id: gain}).

    Formula:
        DCG@K = sum_{i=1..K} ( (2^{rel_i} - 1) / log2(i + 1) )
        IDCG@K = Ideal DCG@K (calculated from sorting ideal gains descending)
        nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0:
        return 0.0

    # Build relevance gain map
    gain_map: Dict[str, float] = {}
    if isinstance(relevance_scores, dict):
        gain_map = {str(item_id): float(gain) for item_id, gain in relevance_scores.items()}
    else:
        gain_map = {str(item_id): 1.0 for item_id in relevance_scores}

    if not gain_map or all(gain <= 0.0 for gain in gain_map.values()):
        return 0.0

    retrieved = _deduplicate_preserve_order(retrieved_ids)[:k]

    # Compute DCG@K
    dcg = 0.0
    for idx, item in enumerate(retrieved, start=1):
        gain = gain_map.get(item, 0.0)
        if gain > 0.0:
            dcg += (math.pow(2.0, gain) - 1.0) / math.log2(idx + 1.0)

    # Compute Ideal DCG@K
    sorted_gains = sorted(gain_map.values(), reverse=True)[:k]
    idcg = 0.0
    for idx, gain in enumerate(sorted_gains, start=1):
        if gain > 0.0:
            idcg += (math.pow(2.0, gain) - 1.0) / math.log2(idx + 1.0)

    if idcg <= 0.0:
        return 0.0

    return float(min(1.0, max(0.0, dcg / idcg)))


def compute_all_retrieval_metrics(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Set[Any], Sequence[Any], Dict[Any, float]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Computes complete retrieval metric suite for a single query.
    """
    relevant_set = (
        set(relevant_ids.keys()) if isinstance(relevant_ids, dict) else set(relevant_ids)
    )
    p_k = precision_at_k(retrieved_ids, relevant_set, k=k)
    r_k = recall_at_k(retrieved_ids, relevant_set, k=k)
    f1_k = f1_at_k(retrieved_ids, relevant_set, k=k)
    hit_k = hit_rate_at_k(retrieved_ids, relevant_set, k=k)
    rr = reciprocal_rank(retrieved_ids, relevant_set, k=k)
    ap = average_precision(retrieved_ids, relevant_set, k=k)
    ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=k)

    return {
        f"precision@{k}": round(p_k, 4),
        f"recall@{k}": round(r_k, 4),
        f"f1@{k}": round(f1_k, 4),
        f"hit_rate@{k}": round(hit_k, 4),
        "reciprocal_rank": round(rr, 4),
        "average_precision": round(ap, 4),
        f"ndcg@{k}": round(ndcg, 4),
    }


def graded_precision_at_k(
    retrieved_ids: Sequence[Any],
    relevance_scores: Dict[Any, float],
    k: int = 5,
    threshold: float = 1.0,
) -> float:
    """
    Precision@K over graded relevance labels where score >= threshold is considered relevant.
    """
    if k <= 0:
        return 0.0
    relevant_ids = {str(item_id) for item_id, score in relevance_scores.items() if float(score) >= threshold}
    return precision_at_k(retrieved_ids, relevant_ids, k=k)


def graded_recall_at_k(
    retrieved_ids: Sequence[Any],
    relevance_scores: Dict[Any, float],
    k: int = 5,
    threshold: float = 1.0,
) -> float:
    """
    Recall@K over graded relevance labels where score >= threshold is considered relevant.
    """
    if k <= 0:
        return 0.0
    relevant_ids = {str(item_id) for item_id, score in relevance_scores.items() if float(score) >= threshold}
    return recall_at_k(retrieved_ids, relevant_ids, k=k)


def evaluate_unanswerable_metrics(
    is_answerable: bool,
    is_refusal: bool,
    total_claims: int = 0,
    supported_claims: int = 0,
    unsupported_claims: int = 0,
) -> Dict[str, float]:
    """
    Computes rigorous unanswerable and refusal metrics:
    - correct_refusal: 1.0 if question is unanswerable and system correctly refused
    - false_answer: 1.0 if question is unanswerable but system generated an answer (hallucination)
    - over_refusal: 1.0 if question is answerable but system erroneously refused
    - unsupported_assertion_rate: fraction of generated claims that lack grounding in context
    """
    correct_refusal = 1.0 if (not is_answerable and is_refusal) else 0.0
    false_answer = 1.0 if (not is_answerable and not is_refusal) else 0.0
    over_refusal = 1.0 if (is_answerable and is_refusal) else 0.0
    unsupported_rate = (
        float(unsupported_claims) / float(total_claims)
        if total_claims > 0
        else (1.0 if false_answer > 0 else 0.0)
    )

    return {
        "correct_refusal": correct_refusal,
        "false_answer": false_answer,
        "over_refusal": over_refusal,
        "unsupported_assertion_rate": round(unsupported_rate, 4),
    }
