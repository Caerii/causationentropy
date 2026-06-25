"""Integration-test benchmark matrix: maps pytest names to discovery presets."""

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
    test_name: str
    preset: str
    data_source: DataSource
    min_tpr: float
    max_fpr: float
    n_nodes: int
    T: int
    seed: int
    rho: float
    p: float


# Each row mirrors a test in test_data_integration.py and its matching preset.
INTEGRATION_BENCHMARKS: List[IntegrationBenchmark] = [
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
    """Return sorted integration test names in the benchmark matrix."""
    return sorted(_BENCHMARK_BY_TEST.keys())


def get_integration_benchmark(test_name: str) -> IntegrationBenchmark:
    """Return the benchmark row for a pytest integration test name."""
    if test_name not in _BENCHMARK_BY_TEST:
        available = ", ".join(list_integration_benchmarks())
        raise KeyError(
            f"Unknown integration test {test_name!r}. Known tests: {available}"
        )
    return dict(_BENCHMARK_BY_TEST[test_name])


def preset_for_integration_test(test_name: str) -> str:
    """Return the discovery preset name for an integration test."""
    return get_integration_benchmark(test_name)["preset"]


def integration_matrix_table() -> str:
    """Return a markdown table of integration tests and presets."""
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
    """Generate synthetic data and ground-truth graph for a benchmark row."""
    G_true = _make_graph(benchmark)
    if benchmark["data_source"] == "poisson":
        data, _ = poisson_coupled_oscillators(
            n=benchmark["n_nodes"],
            T=benchmark["T"],
            seed=benchmark["seed"],
            G=G_true,
        )
    else:
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
    """Run discovery for a named integration test and return recovery metrics."""
    benchmark = get_integration_benchmark(test_name)
    preset = benchmark["preset"]
    get_discovery_preset(preset)  # validate preset exists

    data, G_true = make_integration_data(benchmark)
    kwargs: Dict[str, Any] = dict(
        preset=preset,
        seed=benchmark["seed"],
        show_progress=show_progress,
    )
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs

    G_discovered = discover_network(data, **kwargs)

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
    """Run an integration benchmark and assert TPR/FPR thresholds."""
    result = run_integration_benchmark(
        test_name, show_progress=show_progress, n_jobs=n_jobs
    )
    label = f"{result['preset']} ({test_name})"
    print(f"{label}: TPR={result['tpr']}, FPR={result['fpr']}")
    assert result["tpr"] >= result["min_tpr"]
    assert result["fpr"] <= result["max_fpr"]
    return result
