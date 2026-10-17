import pytest
from app.evaluation.retrieval.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    compute_all_retrieval_metrics,
)


def test_precision_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}

    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5  # 0.4
    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3  # ~0.3333
    assert precision_at_k(retrieved, relevant, k=1) == 0.0
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, set(), k=5) == 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_recall_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4", "c6", "c7"}  # 4 total relevant items

    assert recall_at_k(retrieved, relevant, k=5) == 2 / 4  # 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 1 / 4  # 0.25
    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    # Empty relevant with custom policy
    assert recall_at_k(retrieved, set(), k=5, no_relevant_policy=1.0) == 1.0
    assert recall_at_k(retrieved, set(), k=5, no_relevant_policy=0.0) == 0.0


def test_f1_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}  # P@5 = 2/5 = 0.4, R@5 = 2/2 = 1.0
    # F1 = 2 * (0.4 * 1.0) / (0.4 + 1.0) = 0.8 / 1.4 = 4/7 = 0.5714
    f1 = f1_at_k(retrieved, relevant, k=5)
    assert round(f1, 4) == round(4.0 / 7.0, 4)

    # Zero precision/recall
    assert f1_at_k(retrieved, {"c99"}, k=5) == 0.0
    assert f1_at_k([], relevant, k=5) == 0.0


def test_hit_rate_at_k():
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c2"}

    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0
    assert hit_rate_at_k(retrieved, relevant, k=5) == 1.0
    assert hit_rate_at_k(retrieved, {"c99"}, k=5) == 0.0
    assert hit_rate_at_k([], relevant, k=5) == 0.0


def test_reciprocal_rank_and_mrr():
    q1 = reciprocal_rank(["c1", "c2", "c3"], {"c2"})  # rank 2 -> 0.5
    q2 = reciprocal_rank(["c1", "c2", "c3"], {"c1"})  # rank 1 -> 1.0
    q3 = reciprocal_rank(["c1", "c2", "c3"], {"c9"})  # not found -> 0.0

    assert q1 == 0.5
    assert q2 == 1.0
    assert q3 == 0.0

    mrr = mean_reciprocal_rank([q1, q2, q3])
    assert round(mrr, 4) == round((0.5 + 1.0 + 0.0) / 3, 4)  # 0.5


def test_average_precision_and_map():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}

    # rank 2: P@2 = 1/2
    # rank 4: P@4 = 2/4 = 1/2
    # AP = (0.5 + 0.5) / 2 = 0.5
    ap = average_precision(retrieved, relevant, k=5)
    assert ap == 0.5

    map_val = mean_average_precision([0.5, 1.0, 0.25])
    assert round(map_val, 4) == round(1.75 / 3, 4)


def test_ndcg_at_k_binary_and_graded():
    # 1. Binary relevance
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}
    score = ndcg_at_k(retrieved, relevant, k=5)
    # Expected approx 0.6509 based on standard DCG calculation
    assert 0.65 <= score <= 0.66

    # 2. Perfect ranking should have nDCG = 1.0
    perfect_retrieved = ["c2", "c4", "c1", "c3", "c5"]
    assert ndcg_at_k(perfect_retrieved, relevant, k=2) == 1.0
    assert ndcg_at_k(perfect_retrieved, relevant, k=5) == 1.0

    # 3. Graded relevance
    graded_scores = {"c1": 1.0, "c2": 3.0, "c3": 2.0}
    # Best possible: c2 (3.0), c3 (2.0), c1 (1.0)
    assert ndcg_at_k(["c2", "c3", "c1"], graded_scores, k=3) == 1.0
    # Worst possible: reversed
    imperfect = ndcg_at_k(["c1", "c3", "c2"], graded_scores, k=3)
    assert 0.0 < imperfect < 1.0


def test_duplicate_retrieved_ids_handled_safely():
    # Duplicates in retrieved list shouldn't artificially inflate metrics
    retrieved_with_dups = ["c1", "c1", "c2", "c2", "c3"]
    relevant = {"c2"}

    # Deduplicated is ["c1", "c2", "c3"] -> c2 is at rank 2
    assert reciprocal_rank(retrieved_with_dups, relevant, k=5) == 0.5
    assert precision_at_k(retrieved_with_dups, relevant, k=3) == 1 / 3
    assert recall_at_k(retrieved_with_dups, relevant, k=3) == 1.0


def test_compute_all_retrieval_metrics():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}

    metrics = compute_all_retrieval_metrics(retrieved, relevant, k=5)
    assert metrics["precision@5"] == 0.4
    assert metrics["recall@5"] == 1.0
    assert metrics["hit_rate@5"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["average_precision"] == 0.5
    assert 0.65 <= metrics["ndcg@5"] <= 0.66
