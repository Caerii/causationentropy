#!/usr/bin/env python3
"""Exploration and profiling harness for causal discovery.

This script complements :mod:`examples.benchmark_matrix` by exercising broader
scenarios: the pytest integration matrix, paper-style method grids, scaling
curves, edge cases, and optional ``cProfile`` snapshots.

Examples
--------
Run the integration matrix (same rows as ``test_data_integration.py``)::

    uv run python examples/explore_profile.py --integration

Quick smoke pass (integration + a short reproduction grid)::

    uv run python examples/explore_profile.py --quick

Full exploration including scaling, edge cases, and profiles (slow)::

    uv run python examples/explore_profile.py --all

List registered integration tests and their presets::

    uv run python examples/explore_profile.py --list
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd

from causationentropy import discover_network
from causationentropy.core.integration_benchmarks import (
    integration_matrix_table,
    list_integration_benchmarks,
    run_integration_benchmark,
)
from causationentropy.core.presets import list_presets
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    logisic_dynamics,
    poisson_coupled_oscillators,
)
from causationentropy.graph import evaluate_network_recovery


@dataclass
class Result:
    """One row of exploration output."""

    name: str
    ok: bool
    elapsed_s: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call ``fn`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_integration_suite(
    *,
    n_jobs: int | None,
    quiet: bool,
    test_names: list[str] | None = None,
) -> list[Result]:
    """Exercise every row of the integration benchmark matrix."""
    names = test_names or list_integration_benchmarks()
    results: list[Result] = []

    for test_name in names:
        try:
            t0 = time.perf_counter()
            row = run_integration_benchmark(
                test_name,
                show_progress=not quiet,
                n_jobs=n_jobs,
            )
            elapsed = time.perf_counter() - t0
            passed = row["tpr"] >= row["min_tpr"] and row["fpr"] <= row["max_fpr"]
            results.append(
                Result(
                    f"integration/{test_name}",
                    passed,
                    elapsed,
                    {
                        "preset": row["preset"],
                        "tpr": row["tpr"],
                        "fpr": row["fpr"],
                        "min_tpr": row["min_tpr"],
                        "max_fpr": row["max_fpr"],
                    },
                )
            )
        except Exception as exc:
            results.append(
                Result(
                    f"integration/{test_name}",
                    False,
                    0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return results


def run_reproduction_suite(seed: int = 42, p: float = 0.2) -> list[Result]:
    """Paper-style method × information grid on synthetic benchmarks."""
    results: list[Result] = []

    configs = [
        ("standard+gaussian", {"method": "standard", "information": "gaussian"}),
        ("alternative+gaussian", {"method": "alternative", "information": "gaussian"}),
        ("standard+knn", {"method": "standard", "information": "knn"}),
        ("standard+geometric_knn", {"method": "standard", "information": "geometric_knn"}),
        ("standard+kde", {"method": "standard", "information": "kde"}),
        ("standard+poisson", {"method": "standard", "information": "poisson"}),
        ("lasso", {"method": "lasso", "information": "gaussian"}),
        ("information_lasso", {"method": "information_lasso", "information": "gaussian"}),
    ]

    for label, kwargs in configs:
        if "poisson" in label:
            continue
        try:
            G_true = nx.erdos_renyi_graph(5, p, seed=seed, directed=True)
            data, _ = linear_stochastic_gaussian_process(
                rho=0.7, n=5, T=200, seed=seed, G=G_true
            )
            G, elapsed = timed(
                discover_network,
                data,
                max_lag=1,
                n_shuffles=200,
                alpha_forward=0.05,
                alpha_backward=0.05,
                seed=seed,
                show_progress=False,
                **kwargs,
            )
            tpr, fpr = evaluate_network_recovery(G_true, G)
            results.append(
                Result(
                    f"repro/linear_gaussian/{label}",
                    True,
                    elapsed,
                    {"tpr": tpr, "fpr": fpr, "edges": G.number_of_edges()},
                )
            )
        except Exception as exc:
            results.append(
                Result(
                    f"repro/linear_gaussian/{label}",
                    False,
                    0.0,
                    error=str(exc),
                )
            )

    try:
        G_true = nx.erdos_renyi_graph(5, p, seed=seed, directed=True)
        data, _ = poisson_coupled_oscillators(n=5, T=200, seed=seed, G=G_true)
        for method in ("standard", "alternative"):
            G, elapsed = timed(
                discover_network,
                data,
                method=method,
                information="poisson",
                max_lag=1,
                n_shuffles=200,
                seed=seed,
                show_progress=False,
            )
            tpr, fpr = evaluate_network_recovery(G_true, G)
            results.append(
                Result(
                    f"repro/poisson/{method}",
                    True,
                    elapsed,
                    {"tpr": tpr, "fpr": fpr, "edges": G.number_of_edges()},
                )
            )
    except Exception as exc:
        results.append(Result("repro/poisson/*", False, 0.0, error=str(exc)))

    try:
        data, A = logisic_dynamics(n=5, t=200, seed=seed)
        G_true = nx.from_numpy_array(A, create_using=nx.DiGraph)
        G, elapsed = timed(
            discover_network,
            data,
            preset="logistic_chaos",
            seed=seed,
            show_progress=False,
        )
        tpr, fpr = evaluate_network_recovery(G_true, G)
        results.append(
            Result(
                "repro/logistic/logistic_chaos",
                True,
                elapsed,
                {"tpr": tpr, "fpr": fpr, "edges": G.number_of_edges()},
            )
        )
    except Exception as exc:
        results.append(
            Result("repro/logistic/logistic_chaos", False, 0.0, error=str(exc))
        )

    return results


def run_scaling_suite(seed: int = 42) -> list[Result]:
    """Scale ``n``, ``T``, ``max_lag``, and ``n_shuffles`` (Gaussian standard)."""
    results: list[Result] = []
    base = dict(
        method="standard",
        information="gaussian",
        n_shuffles=50,
        max_lag=1,
        seed=seed,
        show_progress=False,
    )

    for n in (3, 5, 8, 10):
        G = nx.erdos_renyi_graph(n, 0.3, seed=seed, directed=True)
        data, _ = linear_stochastic_gaussian_process(
            rho=0.7, n=n, T=200, seed=seed, G=G
        )
        try:
            _, elapsed = timed(discover_network, data, **base)
            results.append(Result(f"scale/n={n}", True, elapsed, {"n": n, "T": 200}))
        except Exception as exc:
            results.append(Result(f"scale/n={n}", False, 0.0, error=str(exc)))

    for T in (100, 200, 500, 1000):
        G = nx.erdos_renyi_graph(5, 0.2, seed=seed, directed=True)
        data, _ = linear_stochastic_gaussian_process(
            rho=0.7, n=5, T=T, seed=seed, G=G
        )
        _, elapsed = timed(discover_network, data, **base)
        results.append(Result(f"scale/T={T}", True, elapsed, {"T": T}))

    for max_lag in (1, 2, 3, 5):
        G = nx.erdos_renyi_graph(5, 0.2, seed=seed, directed=True)
        data, _ = linear_stochastic_gaussian_process(
            rho=0.7, n=5, T=300, seed=seed, G=G
        )
        _, elapsed = timed(discover_network, data, **{**base, "max_lag": max_lag})
        results.append(
            Result(f"scale/max_lag={max_lag}", True, elapsed, {"max_lag": max_lag})
        )

    for n_shuffles in (10, 50, 100, 200, 500):
        G = nx.erdos_renyi_graph(5, 0.2, seed=seed, directed=True)
        data, _ = linear_stochastic_gaussian_process(
            rho=0.7, n=5, T=200, seed=seed, G=G
        )
        _, elapsed = timed(
            discover_network, data, **{**base, "n_shuffles": n_shuffles}
        )
        results.append(
            Result(
                f"scale/n_shuffles={n_shuffles}",
                True,
                elapsed,
                {"n_shuffles": n_shuffles},
            )
        )

    return results


def run_edge_cases(seed: int = 42) -> list[Result]:
    """Stress inputs that should complete or fail gracefully."""
    results: list[Result] = []
    rng = np.random.default_rng(seed)

    cases: list[tuple[str, Callable[[], Any], dict[str, Any]]] = [
        ("dataframe_input", lambda: pd.DataFrame(rng.normal(size=(100, 3))), {}),
        (
            "constant_column",
            lambda: np.column_stack([np.ones(100), rng.normal(size=(100, 2))]),
            {},
        ),
        ("n=2_minimal", lambda: rng.normal(size=(50, 2)), {}),
        ("short_T_vs_max_lag", lambda: rng.normal(size=(8, 3)), {"max_lag": 5}),
        ("single_variable", lambda: rng.normal(size=(100, 1)), {}),
        ("high_dim_n=15", lambda: rng.normal(size=(200, 15)), {}),
    ]

    for name, data_fn, extra in cases:
        try:
            data = data_fn()
            G, elapsed = timed(
                discover_network,
                data,
                max_lag=extra.get("max_lag", 2),
                n_shuffles=20,
                seed=seed,
                show_progress=False,
            )
            results.append(
                Result(
                    f"edge/{name}",
                    True,
                    elapsed,
                    {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
                )
            )
        except Exception as exc:
            results.append(
                Result(
                    f"edge/{name}",
                    False,
                    0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    G = nx.erdos_renyi_graph(5, 0.2, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=0.7, n=5, T=200, seed=seed, G=G
    )
    G_lasso, _ = timed(
        discover_network, data, method="lasso", max_lag=1, seed=seed, show_progress=False
    )
    G_info, _ = timed(
        discover_network,
        data,
        method="information_lasso",
        max_lag=1,
        seed=seed,
        show_progress=False,
    )
    results.append(
        Result(
            "edge/lasso_vs_information_lasso",
            True,
            0.0,
            {
                "identical_edges": set(G_lasso.edges()) == set(G_info.edges()),
                "lasso_edges": G_lasso.number_of_edges(),
                "info_lasso_edges": G_info.number_of_edges(),
            },
        )
    )

    return results


def profile_discovery(
    *,
    information: str = "gaussian",
    max_lag: int = 2,
    n_shuffles: int = 100,
    seed: int = 42,
) -> str:
    """Return a ``cProfile`` summary for one ``discover_network`` call."""
    G = nx.erdos_renyi_graph(5, 0.2, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=0.7, n=5, T=200, seed=seed, G=G
    )

    pr = cProfile.Profile()
    pr.enable()
    discover_network(
        data,
        method="standard",
        information=information,
        max_lag=max_lag,
        n_shuffles=n_shuffles,
        seed=seed,
        show_progress=False,
    )
    pr.disable()
    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("cumulative").print_stats(25)
    return stream.getvalue()


def print_results(title: str, results: list[Result]) -> None:
    """Pretty-print a block of exploration results."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    for row in results:
        if row.ok:
            detail = ", ".join(f"{k}={v}" for k, v in row.details.items())
            print(f"  OK   {row.name:42s} {row.elapsed_s:7.2f}s  {detail}")
        else:
            print(f"  FAIL {row.name:42s}  {row.error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run the pytest integration benchmark matrix.",
    )
    parser.add_argument(
        "--repro",
        action="store_true",
        help="Run paper-style method × information reproduction grid.",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run Gaussian scaling sweeps (n, T, max_lag, n_shuffles).",
    )
    parser.add_argument(
        "--edge",
        action="store_true",
        help="Run edge-case inputs.",
    )
    parser.add_argument(
        "--profile",
        choices=("none", "gaussian", "knn", "both"),
        default="none",
        help="Optional cProfile snapshots (slow).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Integration matrix plus reproduction grid only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Every suite including scaling, edge cases, and profiles.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print integration test names and discovery presets, then exit.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Print the integration benchmark matrix as Markdown.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for integration runs (-1 = all cores).",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.list:
        print("Integration tests:")
        for name in list_integration_benchmarks():
            print(f"  {name}")
        print("\nDiscovery presets:")
        for preset in list_presets():
            print(f"  {preset}")
        return 0

    if args.matrix:
        print(integration_matrix_table())
        return 0

    run_integration = args.integration or args.quick or args.all
    run_repro = args.repro or args.quick or args.all
    run_scaling = args.scaling or args.all
    run_edge = args.edge or args.all
    profile = args.profile if args.profile != "none" else ("both" if args.all else "none")

    if not any((run_integration, run_repro, run_scaling, run_edge, profile != "none")):
        run_integration = True
        run_repro = True

    print("CausationEntropy exploration profile")
    print(f"Python: {sys.version.split()[0]}")

    n_jobs = None if args.n_jobs == -1 else args.n_jobs
    exit_code = 0

    if run_integration:
        rows = run_integration_suite(n_jobs=n_jobs, quiet=args.quiet)
        print_results("INTEGRATION MATRIX", rows)
        if not all(r.ok for r in rows):
            exit_code = 1

    if run_repro:
        rows = run_reproduction_suite(seed=args.seed)
        print_results("REPRODUCTION (method × information)", rows)
        if not all(r.ok for r in rows):
            exit_code = 1

    if run_scaling:
        rows = run_scaling_suite(seed=args.seed)
        print_results("SCALING (Gaussian standard)", rows)
        if not all(r.ok for r in rows):
            exit_code = 1

    if run_edge:
        rows = run_edge_cases(seed=args.seed)
        print_results("EDGE CASES", rows)
        if not all(r.ok for r in rows):
            exit_code = 1

    if profile in ("gaussian", "both"):
        print("\n" + "=" * 60 + "\nPROFILE: gaussian standard max_lag=2\n" + "=" * 60)
        print(profile_discovery(information="gaussian", max_lag=2, seed=args.seed))

    if profile in ("knn", "both"):
        print("\n" + "=" * 60 + "\nPROFILE: knn standard max_lag=1\n" + "=" * 60)
        print(profile_discovery(information="knn", max_lag=1, n_shuffles=50, seed=args.seed))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
