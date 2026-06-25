"""Integration benchmark matrix for causal discovery reproduction tests.

This module is the single source of truth linking pytest integration test names
to named discovery presets in :mod:`causationentropy.core.presets`. The design
follows a simple principle: every validated reproduction test should be
runnable as ``discover_network(data, preset=...)`` with no hidden parameters.

Each row of :data:`INTEGRATION_BENCHMARKS` records four things the reader needs:

1. Which pytest function exercises the scenario (``test_name``).
2. Which preset encodes the tuned hyperparameters (``preset``).
3. Which synthetic generator matches the information estimator's assumptions
   (``data_source``).
4. What recovery quality the test expects (``min_tpr``, ``max_fpr``).

The helpers :func:`run_integration_benchmark` and :func:`assert_integration_benchmark`
implement the shared workflow that ``test_data_integration.py`` delegates to, so
the test file stays readable while the matrix stays authoritative.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

import networkx as nx
import numpy as np

from causationentropy import discover_network
from causationentropy.core.presets import get_discovery_preset
from causationentropy.core.stats import Compute_TPR_FPR
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    poisson_coupled_oscillators,
)

DataSource = Literal["linear_gaussian", "poisson"]


class IntegrationBenchmark(TypedDict):
    """One row of the integration reproduction matrix.

    Each field is documented here so the matrix doubles as a readable catalog
    of what we validate and how strictly we validate it.
    """

    test_name: str  # pytest name in test_data_integration.py
    preset: str  # key in DISCOVERY_PRESETS
    data_source: DataSource  # synthetic generator family
    min_tpr: float  # minimum true-positive rate (off-diagonal edges)
    max_fpr: float  # maximum false-positive rate
    n_nodes: int  # Erdős–Rényi graph order
    T: int  # time-series length
    seed: int  # RNG seed for reproducibility
    rho: float  # VAR coupling (linear Gaussian data only)
    p: float  # Erdős–Rényi edge probability


# Each row mirrors a test in test_data_integration.py and its matching preset.
# Thresholds (min_tpr, max_fpr) are the acceptance criteria baked into pytest.
INTEGRATION_BENCHMARKS: List[IntegrationBenchmark] = [
    # --- Linear Gaussian: standard and alternative oCSE with closed-form CMI ---
    {
        "test_name": "test_standard_gaussian",
        "preset": "reproduction",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_alternative_gaussian",
        "preset": "gaussian_alternative",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    # --- k-NN CMI: metric and neighbor-count sensitivity ---
    {
        "test_name": "test_standard_knn",
        "preset": "knn_standard",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_alternative_knn",
        "preset": "knn_alternative",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_minkowski_standard_knn",
        "preset": "knn_minkowski",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    # --- Geometric k-NN: manifold-aware entropy estimator ---
    {
        "test_name": "test_standard_geometric_knn",
        "preset": "geometric_knn_standard",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    # --- KDE CMI: bandwidth choice (Silverman vs Scott) ---
    {
        "test_name": "test_standard_kde",
        "preset": "kde_standard",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    # --- Regularized baselines (no full permutation pipeline in selection) ---
    {
        "test_name": "test_information_lasso",
        "preset": "information_lasso",
        "data_source": "linear_gaussian",
        "min_tpr": 0.9,
        "max_fpr": 0.2,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_lasso",
        "preset": "lasso",
        "data_source": "linear_gaussian",
        "min_tpr": 0.9,
        "max_fpr": 0.2,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    # --- Poisson CMI: count-valued coupled oscillators ---
    {
        "test_name": "test_standard_poisson",
        "preset": "poisson_standard",
        "data_source": "poisson",
        "min_tpr": 0.95,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_alternative_poisson",
        "preset": "poisson_alternative",
        "data_source": "poisson",
        "min_tpr": 0.95,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_alternative_geometric_knn",
        "preset": "geometric_knn_alternative",
        "data_source": "linear_gaussian",
        "min_tpr": 0.4,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_alternative_kde",
        "preset": "kde_alternative",
        "data_source": "linear_gaussian",
        "min_tpr": 0.1,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_kde_scott_bandwidth",
        "preset": "kde_scott",
        "data_source": "linear_gaussian",
        "min_tpr": 0.95,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_knn_chebyshev_metric",
        "preset": "knn_chebyshev",
        "data_source": "linear_gaussian",
        "min_tpr": 0.9,
        "max_fpr": 0.1,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
    {
        "test_name": "test_knn_manhattan_metric",
        "preset": "knn_manhattan",
        "data_source": "linear_gaussian",
        "min_tpr": 1.0,
        "max_fpr": 0.0,
        "n_nodes": 5,
        "T": 200,
        "seed": 42,
        "rho": 0.7,
        "p": 0.2,
    },
]

_BENCHMARK_BY_TEST: Dict[str, IntegrationBenchmark] = {
    row["test_name"]: row for row in INTEGRATION_BENCHMARKS
}


def list_integration_benchmarks() -> List[str]:
    """Return sorted pytest names registered in the benchmark matrix."""
    return sorted(_BENCHMARK_BY_TEST.keys())


def get_integration_benchmark(test_name: str) -> IntegrationBenchmark:
    """Look up the full benchmark row for a pytest integration test.

    Parameters
    ----------
    test_name : str
        Name of a function in ``test_data_integration.py`` (for example
        ``"test_standard_gaussian"``).

    Returns
    -------
    IntegrationBenchmark
        A copy of the matrix row, including preset name and acceptance thresholds.

    Raises
    ------
    KeyError
        If ``test_name`` is not registered in :data:`INTEGRATION_BENCHMARKS`.
    """
    if test_name not in _BENCHMARK_BY_TEST:
        available = ", ".join(list_integration_benchmarks())
        raise KeyError(
            f"Unknown integration test {test_name!r}. Known tests: {available}"
        )
    return dict(_BENCHMARK_BY_TEST[test_name])


def preset_for_integration_test(test_name: str) -> str:
    """Return the discovery preset that reproduces a given integration test."""
    return get_integration_benchmark(test_name)["preset"]


def integration_matrix_table() -> str:
    """Format the benchmark matrix as a Markdown table for docs and logs."""
    lines = [
        "| Integration test | Preset | Data | Min TPR | Max FPR |",
        "|---|---|---|---:|---:|",
    ]
    for row in INTEGRATION_BENCHMARKS:
        lines.append(
            f"| `{row['test_name']}` | `{row['preset']}` | "
            f"{row['data_source']} | {row['min_tpr']} | {row['max_fpr']} |"
        )
    return "\n".join(lines)


def _make_graph(benchmark: IntegrationBenchmark) -> nx.DiGraph:
    """Draw a fixed Erdős–Rényi ground-truth graph for reproducible benchmarks."""
    np.random.seed(benchmark["seed"])
    return nx.erdos_renyi_graph(
        benchmark["n_nodes"],
        benchmark["p"],
        seed=benchmark["seed"],
        directed=True,
    )


def make_integration_data(
    benchmark: IntegrationBenchmark,
) -> tuple[np.ndarray, nx.DiGraph]:
    """Generate synthetic time series and its ground-truth causal graph.

    The generator is chosen from ``benchmark["data_source"]`` so that the
    distributional assumptions of the information estimator are respected:
    linear Gaussian VAR for ``gaussian``/``knn``/``kde``/``geometric_knn``,
    Poisson counts for ``poisson``.
    """
    G_true = _make_graph(benchmark)
    if benchmark["data_source"] == "poisson":
        # Count-valued dynamics: Poisson coupled oscillators on the same graph.
        data, _ = poisson_coupled_oscillators(
            n=benchmark["n_nodes"],
            T=benchmark["T"],
            seed=benchmark["seed"],
            G=G_true,
        )
    else:
        # Continuous linear stochastic process on the same random graph.
        data, _ = linear_stochastic_gaussian_process(
            rho=benchmark["rho"],
            n=benchmark["n_nodes"],
            T=benchmark["T"],
            seed=benchmark["seed"],
            G=G_true,
        )
    return data, G_true


def run_integration_benchmark(
    test_name: str,
    *,
    show_progress: bool = False,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run causal discovery for one matrix row and measure graph recovery.

    This is the programmatic counterpart to a single pytest integration test:
    load the preset, generate data, call :func:`discover_network`, and compare
    the discovered adjacency to ground truth via TPR/FPR.

    Returns
    -------
    dict
        Keys include ``test_name``, ``preset``, ``tpr``, ``fpr``, and the
        acceptance thresholds ``min_tpr`` / ``max_fpr`` for convenience.
    """
    benchmark = get_integration_benchmark(test_name)
    preset = benchmark["preset"]
    get_discovery_preset(preset)  # fail fast if preset name drifts from matrix

    data, G_true = make_integration_data(benchmark)
    kwargs: Dict[str, Any] = dict(
        preset=preset,
        seed=benchmark["seed"],
        show_progress=show_progress,
    )
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs

    G_discovered = discover_network(data, **kwargs)

    # Binarize adjacency for TPR/FPR (multi-edges across lags collapse to presence).
    A_bin = (nx.to_numpy_array(G_true) > 0).astype(int)
    B_bin = (nx.to_numpy_array(G_discovered) > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)

    return {
        "test_name": test_name,
        "preset": preset,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "min_tpr": benchmark["min_tpr"],
        "max_fpr": benchmark["max_fpr"],
    }


def assert_integration_benchmark(
    test_name: str,
    *,
    show_progress: bool = False,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run a benchmark and assert that recovery meets the matrix thresholds.

    Pytest integration tests call this wrapper so failure messages include
    the preset name and measured TPR/FPR in one line of output.
    """
    result = run_integration_benchmark(
        test_name, show_progress=show_progress, n_jobs=n_jobs
    )
    label = f"{result['preset']} ({test_name})"
    print(f"{label}: TPR={result['tpr']}, FPR={result['fpr']}")
    assert result["tpr"] >= result["min_tpr"]
    assert result["fpr"] <= result["max_fpr"]
    return result
