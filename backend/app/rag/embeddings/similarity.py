import math
from typing import List, Union

try:
    import numpy as np
except ImportError:
    np = None


def validate_vectors(u: List[float], v: List[float]) -> None:
    """Validates vector dimensions and finite numeric values."""
    if len(u) != len(v):
        raise ValueError(
            f"Dimension mismatch: vector u has dimension {len(u)}, while vector v has dimension {len(v)}."
        )
    if len(u) == 0:
        raise ValueError("Cannot calculate similarity on empty vectors.")

    for val in u:
        if not math.isfinite(val):
            raise ValueError("Vector u contains non-finite values (NaN or Inf).")
    for val in v:
        if not math.isfinite(val):
            raise ValueError("Vector v contains non-finite values (NaN or Inf).")


def dot_product(u: List[float], v: List[float]) -> float:
    """
    Computes the dot product (inner product) between two vectors:
    dot(u, v) = sum(u_i * v_i)
    """
    validate_vectors(u, v)
    if np is not None:
        return float(np.dot(np.array(u, dtype=float), np.array(v, dtype=float)))
    return sum(x * y for x, y in zip(u, v))


def euclidean_distance(u: List[float], v: List[float]) -> float:
    """
    Computes the Euclidean distance (L2 norm of the difference):
    d(u, v) = sqrt(sum((u_i - v_i)^2))

    Note: Euclidean distance is a distance metric, not a similarity score.
    0.0 indicates identity, larger numbers indicate greater geometric separation.
    """
    validate_vectors(u, v)
    if np is not None:
        diff = np.array(u, dtype=float) - np.array(v, dtype=float)
        return float(np.linalg.norm(diff))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(u, v)))


def cosine_similarity(u: List[float], v: List[float], eps: float = 1e-12) -> float:
    """
    Computes the Cosine Similarity between two vectors:
    cos_sim(u, v) = dot(u, v) / (||u||_2 * ||v||_2)

    Handles zero vectors safely by returning 0.0 without division by zero.
    Bounded in [-1.0, 1.0].
    """
    validate_vectors(u, v)

    if np is not None:
        arr_u = np.array(u, dtype=float)
        arr_v = np.array(v, dtype=float)
        norm_u = float(np.linalg.norm(arr_u))
        norm_v = float(np.linalg.norm(arr_v))
        if norm_u < eps or norm_v < eps:
            return 0.0
        val = float(np.dot(arr_u, arr_v) / (norm_u * norm_v))
        return float(np.clip(val, -1.0, 1.0))

    norm_u = math.sqrt(sum(x * x for x in u))
    norm_v = math.sqrt(sum(x * x for x in v))
    if norm_u < eps or norm_v < eps:
        return 0.0

    raw_cos = sum(x * y for x, y in zip(u, v)) / (norm_u * norm_v)
    return max(-1.0, min(1.0, raw_cos))


def pairwise_similarity_matrix(
    vectors: List[List[float]],
    metric: str = "cosine",
) -> List[List[float]]:
    """
    Computes pairwise similarity or distance matrix for a list of vectors.
    Supported metrics: 'cosine', 'dot_product', 'euclidean'.
    """
    if not vectors:
        return []

    n = len(vectors)
    dim = len(vectors[0])
    for idx, vec in enumerate(vectors):
        if len(vec) != dim:
            raise ValueError(
                f"Inconsistent vector dimension at index {idx}: expected {dim}, got {len(vec)}."
            )

    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

    metric_lower = metric.lower()
    for i in range(n):
        for j in range(i, n):
            if metric_lower == "cosine":
                val = 1.0 if i == j else cosine_similarity(vectors[i], vectors[j])
            elif metric_lower == "dot_product":
                val = dot_product(vectors[i], vectors[j])
            elif metric_lower in ("euclidean", "euclidean_distance"):
                val = 0.0 if i == j else euclidean_distance(vectors[i], vectors[j])
            else:
                raise ValueError(
                    f"Unknown similarity metric: '{metric}'. Supported: 'cosine', 'dot_product', 'euclidean'."
                )

            val_rounded = round(float(val), 6)
            matrix[i][j] = val_rounded
            matrix[j][i] = val_rounded

    return matrix
