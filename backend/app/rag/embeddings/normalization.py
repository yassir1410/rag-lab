import math
from typing import List, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None


def compute_l2_norm(vector: List[float]) -> float:
    """Computes the Euclidean L2 norm of a vector: ||v||_2 = sqrt(sum(v_i^2))."""
    if not vector:
        return 0.0
    if np is not None:
        return float(np.linalg.norm(np.array(vector, dtype=float)))
    return math.sqrt(sum(x * x for x in vector))


def l2_normalize(vector: List[float], eps: float = 1e-12) -> List[float]:
    """
    Normalizes a vector to unit length (Euclidean norm = 1.0).
    If the vector has norm <= eps, returns the vector unchanged (or zero vector).
    """
    if not vector:
        return []

    norm = compute_l2_norm(vector)
    if norm <= eps:
        return list(vector)

    if np is not None:
        arr = np.array(vector, dtype=float)
        normalized = arr / max(norm, eps)
        return [float(x) for x in normalized]

    return [x / norm for x in vector]


def inspect_normalization(vector: List[float], tolerance: float = 1e-4) -> Dict[str, Any]:
    """
    Analyzes the normalization state of a vector.
    Returns the initial norm, whether it is approximately unit-normalized, and its normalized counterpart.
    """
    norm = compute_l2_norm(vector)
    is_unit = abs(norm - 1.0) <= tolerance
    normalized_vec = l2_normalize(vector)

    return {
        "original_norm": round(norm, 6),
        "is_unit_length": is_unit,
        "dimension": len(vector),
        "normalized_vector_sample": normalized_vec[:5] if len(normalized_vec) >= 5 else normalized_vec,
    }
