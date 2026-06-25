#!/usr/bin/env python3
"""Run validated reproduction benchmarks using discovery presets.

Examples
--------
Reproduce the primary linear Gaussian benchmark::

    uv run python examples/reproduce_benchmark.py --preset reproduction

List all available presets::

    uv run python examples/reproduce_benchmark.py --list
"""

from __future__ import annotations

import argparse
import time

import networkx as nx

from causationentropy import discover_network
from causationentropy.core.presets import describe_preset, list_presets
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    poisson_coupled_oscillators,
)
from causationentropy.graph import evaluate_network_recovery


def _synthetic_data(preset: str, seed: int, n: int, T: int):
    p = 0.2
    G_true = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    if preset.startswith("poisson"):
        data, _ = poisson_coupled_oscillators(n=n, T=T, seed=seed, G=G_true)
    else:
        data, _ = linear_stochastic_gaussian_process(
            rho=0.7, n=n, T=T, seed=seed, G=G_true
        )
    return data, G_true


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="reproduction",
        help="Named discovery preset (see --list).",
    )
    parser.add_argument("--list", action="store_true", help="List presets and exit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in list_presets():
            print(f"{name:28s} {describe_preset(name)}")
        return 0

    data, G_true = _synthetic_data(args.preset, args.seed, args.nodes, args.length)
    print(f"Preset: {args.preset}")
    print(f"Description: {describe_preset(args.preset)}")
    print(f"Data shape: {data.shape}")

    t0 = time.perf_counter()
    G_disc = discover_network(
        data,
        preset=args.preset,
        seed=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        show_progress=not args.quiet,
    )
    elapsed = time.perf_counter() - t0

    tpr, fpr = evaluate_network_recovery(G_true, G_disc)
    print(f"Runtime: {elapsed:.2f}s")
    print(f"Discovered edges: {G_disc.number_of_edges()}")
    print(f"TPR: {tpr:.3f}  FPR: {fpr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
