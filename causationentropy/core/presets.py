"""Validated discovery presets from integration benchmarks.

Each preset is a named dictionary of ``discover_network`` keyword arguments that
were tuned until the corresponding integration test in
``test_data_integration.py`` met its TPR/FPR gate. Presets exist so users can
reproduce paper-style results without copying long parameter lists, and so the
library can distinguish **interactive defaults** (fast, ``demo``) from
**validated reproduction settings** (``reproduction``, ``knn_standard``, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class DiscoveryPreset(TypedDict, total=False):
    method: str
    information: str
    max_lag: int
    alpha_forward: float
    alpha_backward: float
    n_shuffles: int
    metric: str
    bandwidth: str
    k: int
    n_jobs: int
    description: str


DISCOVERY_PRESETS: Dict[str, DiscoveryPreset] = {
    "reproduction": {
        "method": "standard",
        "information": "gaussian",
        "max_lag": 1,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Primary paper-style linear Gaussian benchmark (TPR=1, FPR=0).",
    },
    "gaussian_standard": {
        "method": "standard",
        "information": "gaussian",
        "max_lag": 1,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard oCSE with Gaussian CMI on linear VAR data.",
    },
    "gaussian_alternative": {
        "method": "alternative",
        "information": "gaussian",
        "max_lag": 1,
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Alternative oCSE with Gaussian CMI on linear VAR data.",
    },
    "knn_standard": {
        "method": "standard",
        "information": "knn",
        "max_lag": 2,
        "metric": "euclidean",
        "k": 5,
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard oCSE with k-NN CMI (tuned for linear Gaussian data).",
    },
    "knn_alternative": {
        "method": "alternative",
        "information": "knn",
        "max_lag": 2,
        "metric": "euclidean",
        "k": 20,
        "alpha_forward": 0.001,
        "alpha_backward": 0.001,
        "n_shuffles": 5000,
        "n_jobs": -1,
        "description": "Alternative oCSE with k-NN CMI (high k, many shuffles).",
    },
    "knn_minkowski": {
        "method": "standard",
        "information": "knn",
        "max_lag": 2,
        "metric": "minkowski",
        "k": 5,
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard k-NN CMI with Minkowski metric.",
    },
    "knn_chebyshev": {
        "method": "standard",
        "information": "knn",
        "max_lag": 2,
        "metric": "chebyshev",
        "k": 8,
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard k-NN CMI with Chebyshev metric.",
    },
    "knn_manhattan": {
        "method": "standard",
        "information": "knn",
        "max_lag": 2,
        "metric": "cityblock",
        "k": 8,
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard k-NN CMI with Manhattan (cityblock) metric.",
    },
    "geometric_knn_standard": {
        "method": "standard",
        "information": "geometric_knn",
        "max_lag": 2,
        "metric": "minkowski",
        "k": 10,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 500,
        "n_jobs": -1,
        "description": "Standard oCSE with geometric k-NN CMI.",
    },
    "geometric_knn_alternative": {
        "method": "alternative",
        "information": "geometric_knn",
        "max_lag": 2,
        "metric": "euclidean",
        "k": 10,
        "alpha_forward": 0.001,
        "alpha_backward": 0.001,
        "n_shuffles": 500,
        "n_jobs": -1,
        "description": "Alternative oCSE with geometric k-NN CMI.",
    },
    "kde_standard": {
        "method": "standard",
        "information": "kde",
        "max_lag": 2,
        "bandwidth": "silverman",
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard oCSE with KDE CMI (Silverman bandwidth).",
    },
    "kde_alternative": {
        "method": "alternative",
        "information": "kde",
        "max_lag": 2,
        "bandwidth": "silverman",
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Alternative oCSE with KDE CMI.",
    },
    "kde_scott": {
        "method": "standard",
        "information": "kde",
        "max_lag": 2,
        "bandwidth": "scott",
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard oCSE with KDE CMI (Scott bandwidth).",
    },
    "poisson_standard": {
        "method": "standard",
        "information": "poisson",
        "max_lag": 1,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Standard oCSE with Poisson CMI on count data.",
    },
    "poisson_alternative": {
        "method": "alternative",
        "information": "poisson",
        "max_lag": 1,
        "alpha_forward": 0.01,
        "alpha_backward": 0.01,
        "n_shuffles": 1000,
        "n_jobs": -1,
        "description": "Alternative oCSE with Poisson CMI on count data.",
    },
    "lasso": {
        "method": "lasso",
        "information": "gaussian",
        "max_lag": 1,
        "n_jobs": 1,
        "description": "LASSO baseline (no permutation testing in selection).",
    },
    "information_lasso": {
        "method": "information_lasso",
        "information": "gaussian",
        "max_lag": 1,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 1000,
        "n_jobs": 1,
        "description": "MI-weighted LASSO screening with backward permutation pruning and high-dimensional screening.",
    },
    "logistic_chaos": {
        "method": "standard",
        "information": "gaussian",
        "max_lag": 1,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 500,
        "n_jobs": -1,
        "description": "Nonlinear logistic-map network discovery (exploratory).",
    },
    "demo": {
        "method": "standard",
        "information": "gaussian",
        "max_lag": 5,
        "alpha_forward": 0.05,
        "alpha_backward": 0.05,
        "n_shuffles": 200,
        "n_jobs": -1,
        "description": "Fast interactive demo; may differ from validated reproduction settings.",
    },
}


def list_presets() -> List[str]:
    """Return sorted preset names."""
    return sorted(DISCOVERY_PRESETS.keys())


def get_discovery_preset(name: str) -> DiscoveryPreset:
    """Return a copy of a named discovery preset."""
    if name not in DISCOVERY_PRESETS:
        available = ", ".join(list_presets())
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}")
    return dict(DISCOVERY_PRESETS[name])


def describe_preset(name: str) -> str:
    """Return the human-readable description for a preset."""
    preset = get_discovery_preset(name)
    return preset.get("description", "")


def apply_preset_to_params(
    preset: str,
    method: str,
    information: str,
    max_lag: int,
    alpha_forward: float,
    alpha_backward: float,
    metric: str,
    bandwidth: str,
    k_neighbors: int,
    n_shuffles: int,
    n_jobs: int,
) -> Dict[str, Any]:
    """Merge a preset into explicit discovery parameters."""
    cfg = get_discovery_preset(preset)
    return {
        "method": cfg.get("method", method),
        "information": cfg.get("information", information),
        "max_lag": cfg.get("max_lag", max_lag),
        "alpha_forward": cfg.get("alpha_forward", alpha_forward),
        "alpha_backward": cfg.get("alpha_backward", alpha_backward),
        "metric": cfg.get("metric", metric),
        "bandwidth": cfg.get("bandwidth", bandwidth),
        "k_neighbors": cfg.get("k", k_neighbors),
        "n_shuffles": cfg.get("n_shuffles", n_shuffles),
        "n_jobs": cfg.get("n_jobs", n_jobs),
    }
