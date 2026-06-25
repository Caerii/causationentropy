#!/usr/bin/env python3
"""Timing matrix over discovery presets on synthetic benchmarks.

Examples
--------
Run a quick subset (demo + reproduction presets)::

    uv run python examples/benchmark_matrix.py --quick

Run all presets (slow; includes poisson and geometric_knn)::

    uv run python examples/benchmark_matrix.py --all
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from causationentropy import discover_network
from causationentropy.core.presets import list_presets
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    logisic_dynamics,
    poisson_coupled_oscillators,
)
from causationentropy.graph import evaluate_network_recovery


@dataclass
class BenchmarkRow:
    preset: str
    ok: bool
    elapsed_s: float
    tpr: float | None = None
    fpr: float | None = None
    edges: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _synthetic_data(preset: str, seed: int, n: int, T: int):
    p = 0.2
    G_true = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    if preset.startswith("poisson"):
        data, _ = poisson_coupled_oscillators(n=n, T=T, seed=seed, G=G_true)
    elif preset.startswith("logistic"):
        data, adjacency = logisic_dynamics(n=n, t=T, seed=seed)
        G_true = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        return data, G_true
    data, _ = linear_stochastic_gaussian_process(
        rho=0.7, n=n, T=T, seed=seed, G=G_true
    )
    return data, G_true


def run_preset(
    preset: str,
    seed: int,
    n: int,
    T: int,
    n_jobs: int,
    quiet: bool,
) -> BenchmarkRow:
    try:
        data, G_true = _synthetic_data(preset, seed, n, T)
        t0 = time.perf_counter()
        G_disc = discover_network(
            data,
            preset=preset,
            seed=seed,
            n_jobs=n_jobs,
            show_progress=not quiet,
        )
        elapsed = time.perf_counter() - t0
        tpr, fpr = evaluate_network_recovery(G_true, G_disc)
        return BenchmarkRow(
            preset=preset,
            ok=True,
            elapsed_s=elapsed,
            tpr=tpr,
            fpr=fpr,
            edges=G_disc.number_of_edges(),
            details={"shape": data.shape},
        )
    except Exception as exc:
        return BenchmarkRow(preset=preset, ok=False, elapsed_s=0.0, error=str(exc))


def _select_presets(quick: bool, all_presets: bool) -> list[str]:
    names = list_presets()
    if all_presets:
        return names
    if quick:
        return [p for p in ("demo", "reproduction", "gaussian_standard", "logistic_chaos") if p in names]
    skip_slow = {"poisson_standard", "poisson_alternative", "knn_alternative"}
    return [p for p in names if p not in skip_slow]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small preset subset.")
    parser.add_argument("--all", action="store_true", help="Run every preset (slow).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    presets = _select_presets(args.quick, args.all)
    print(f"{'preset':28s} {'time(s)':>8s} {'TPR':>6s} {'FPR':>6s} {'edges':>6s}  status")
    print("-" * 72)

    rows: list[BenchmarkRow] = []
    for preset in presets:
        row = run_preset(
            preset, args.seed, args.nodes, args.length, args.n_jobs, args.quiet
        )
        rows.append(row)
        if row.ok:
            print(
                f"{row.preset:28s} {row.elapsed_s:8.2f} {row.tpr:6.3f} {row.fpr:6.3f} "
                f"{row.edges:6d}  OK"
            )
        else:
            print(f"{row.preset:28s} {'—':>8s} {'—':>6s} {'—':>6s} {'—':>6s}  FAIL: {row.error}")

    ok_count = sum(1 for r in rows if r.ok)
    total_time = sum(r.elapsed_s for r in rows if r.ok)
    print("-" * 72)
    print(f"{ok_count}/{len(rows)} presets OK  total runtime {total_time:.1f}s")
    return 0 if ok_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
