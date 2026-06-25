"""Distance-matrix and correlation-determinant caches for information estimators."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from causationentropy.core.linalg import correlation_log_determinant

_distance_cache: Dict[str, np.ndarray] = {}
_tree_cache: Dict[str, Any] = {}
_detcorr_cache: Dict[str, float] = {}

_corrcoef_cache: Dict[str, np.ndarray] = {}

_distance_cache_size = 128
_tree_cache_size = 100
_detcorr_cache_size = 128
_corrcoef_cache_size = 128


def _array_hash(arr: np.ndarray, metric: str = "euclidean", extra: str = "") -> str:
    payload = f"{arr.tobytes()}{metric}{extra}".encode()
    return hashlib.md5(payload).hexdigest()


def set_distance_cache_size(size: int) -> None:
    """Set maximum entries for distance-matrix and spatial-tree caches."""
    global _distance_cache_size, _tree_cache_size
    _distance_cache_size = max(16, int(size))
    _tree_cache_size = max(16, int(size))


def clear_caches() -> None:
    """Clear all estimator caches."""
    _distance_cache.clear()
    _tree_cache.clear()
    _detcorr_cache.clear()
    _corrcoef_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Return current cache occupancy and limits."""
    return {
        "distance_cache_size": len(_distance_cache),
        "distance_cache_limit": _distance_cache_size,
        "tree_cache_size": len(_tree_cache),
        "tree_cache_limit": _tree_cache_size,
        "detcorr_cache_size": len(_detcorr_cache),
        "detcorr_cache_limit": _detcorr_cache_size,
        "corrcoef_cache_size": len(_corrcoef_cache),
        "corrcoef_cache_limit": _corrcoef_cache_size,
    }


def estimate_cache_size(n_vars: int, max_lag: int, n_samples: int) -> Tuple[int, float]:
    """Estimate a reasonable cache size and memory footprint in MB."""
    predictors_per_var = n_vars * max_lag
    operations_per_var = 2 * predictors_per_var
    total_operations = n_vars * operations_per_var
    unique_matrices = min(total_operations * 6, 1000)
    avg_matrix_size = n_samples * n_samples * 8
    memory_mb = unique_matrices * avg_matrix_size / (1024 * 1024)
    if memory_mb < 100:
        cache_size = unique_matrices
    elif memory_mb < 500:
        cache_size = int(unique_matrices * 0.5)
    else:
        cache_size = min(200, int(unique_matrices * 0.2))
    return max(16, cache_size), memory_mb


def configure_cache_for_discovery(
    data_shape: Tuple[int, int],
    max_lag: int = 5,
    information_method: str = "knn",
) -> Dict[str, Any]:
    """Configure cache limits from discovery problem size."""
    n_samples, n_vars = data_shape
    cache_size, memory_mb = estimate_cache_size(n_vars, max_lag, n_samples)
    if information_method in ("knn", "geometric_knn"):
        multiplier = 1.0
    elif information_method == "gaussian":
        multiplier = 0.3
    elif information_method == "kde":
        multiplier = 0.6
    else:
        multiplier = 0.5
    adjusted = max(16, min(1000, int(cache_size * multiplier)))
    set_distance_cache_size(adjusted)
    clear_caches()
    return {
        "cache_size": adjusted,
        "estimated_memory_mb": memory_mb * multiplier,
        "information_method": information_method,
    }


def cached_detcorr(A: np.ndarray) -> float:
    """Cached signed log-determinant of a correlation matrix."""
    A = np.asarray(A)
    if A.shape[1] == 0:
        return 0.0
    key = _array_hash(A, extra="detcorr")
    if key in _detcorr_cache:
        return _detcorr_cache[key]
    result = correlation_log_determinant(A)
    if len(_detcorr_cache) >= _detcorr_cache_size:
        del _detcorr_cache[next(iter(_detcorr_cache))]
    _detcorr_cache[key] = result
    return result


def _correlation_log_det_from_matrix(C: np.ndarray) -> float:
    """Signed log-determinant of a correlation matrix with singular fallback."""
    if C.ndim == 0:
        return 0.0
    sign, logdet = np.linalg.slogdet(C)
    if sign == 0 or not np.isfinite(logdet):
        return -1000.0
    return float(logdet)


def cached_corrcoef(A: np.ndarray) -> np.ndarray:
    """Cached full correlation matrix for a data block."""
    A = np.asarray(A)
    if A.shape[1] == 0:
        return np.zeros((0, 0))
    key = _array_hash(A, extra="corrcoef")
    if key in _corrcoef_cache:
        return _corrcoef_cache[key]
    C = np.corrcoef(A.T)
    if len(_corrcoef_cache) >= _corrcoef_cache_size:
        del _corrcoef_cache[next(iter(_corrcoef_cache))]
    _corrcoef_cache[key] = C
    return C


def correlation_log_det_subset(C: np.ndarray, indices: np.ndarray) -> float:
    """Log-determinant of a correlation submatrix by column indices."""
    if indices.size == 0:
        return 0.0
    if indices.size == 1:
        return 0.0
    sub = C[np.ix_(indices, indices)]
    return _correlation_log_det_from_matrix(sub)


def cached_cdist(
    data: np.ndarray,
    metric: str = "euclidean",
    p: float = 2.0,
) -> np.ndarray:
    """Cached full pairwise distance matrix."""
    data = np.asarray(data)
    key = _array_hash(data, metric=metric, extra=str(p))
    cached = _distance_cache.get(key)
    if cached is not None and cached.shape[0] == data.shape[0]:
        return cached
    if metric == "minkowski":
        result = cdist(data, data, metric=metric, p=p)
    else:
        result = cdist(data, data, metric=metric)
    if len(_distance_cache) >= _distance_cache_size:
        del _distance_cache[next(iter(_distance_cache))]
    _distance_cache[key] = result
    return result


def get_or_build_tree(data: np.ndarray, metric: str = "euclidean"):
    """Return a cached KDTree or BallTree for neighbor queries."""
    from sklearn.neighbors import BallTree, KDTree

    data = np.asarray(data)
    key = _array_hash(data, metric=metric, extra="tree")
    if key in _tree_cache:
        return _tree_cache[key]
    _, n_features = data.shape
    if metric == "euclidean" and n_features <= 15:
        tree = KDTree(data, metric=metric)
    else:
        tree = BallTree(data, metric=metric)
    if len(_tree_cache) >= _tree_cache_size:
        del _tree_cache[next(iter(_tree_cache))]
    _tree_cache[key] = tree
    return tree


def tree_knn_distances(
    data: np.ndarray,
    k: int = 1,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return k-nearest-neighbor distances and indices (excluding self)."""
    tree = get_or_build_tree(data, metric=metric)
    distances, indices = tree.query(data, k=k + 1)
    if k == 0:
        return distances[:, :1], indices[:, :1]
    if distances.ndim == 1:
        return distances[None, 1:], indices[None, 1:]
    return distances[:, 1:], indices[:, 1:]


def tree_neighbors_within_distance(
    data: np.ndarray,
    distances: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Count neighbors within per-point distance thresholds."""
    tree = get_or_build_tree(data, metric=metric)
    counts = np.zeros(len(data))
    for i, epsilon in enumerate(distances):
        neighbors = tree.query_radius([data[i]], r=epsilon)[0]
        counts[i] = len(neighbors) - 1
    return counts


def supports_kd_tree(metric: str) -> bool:
    """Return True when tree-based neighbor search is supported for a metric."""
    return metric in {"euclidean", "chebyshev", "cityblock", "manhattan"}
