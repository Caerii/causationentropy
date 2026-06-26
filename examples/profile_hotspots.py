#!/usr/bin/env python3
"""Empirical hotspot report for ``discover_network``.

Runs wall-clock timing and ``cProfile`` on representative presets, then
summarizes where cumulative time is spent. Use this before performance PRs
so upgrades are driven by measurement, not guesswork.

Examples
--------
Default trio at profile tier (``n_shuffles=80``, preset hyperparameters otherwise)::

    uv run python examples/profile_hotspots.py

Full preset ``n_shuffles`` (slow; e.g. reproduction uses 1000)::

    uv run python examples/profile_hotspots.py --tier full --preset reproduction
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np

from causationentropy import discover_network
from causationentropy.core.presets import get_discovery_preset, list_presets
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    poisson_coupled_oscillators,
)


@dataclass
class HotspotRow:
    """One line in the aggregated hotspot table."""

    label: str
    cumulative_s: float
    share_pct: float
    tottime_s: float


@dataclass
class ProfileReport:
    """Wall time plus ranked hotspots for one preset run."""

    preset: str
    wall_s: float
    n_nodes: int
    n_edges: int
    data_shape: tuple[int, int]
    hotspots: list[HotspotRow] = field(default_factory=list)
    top_functions: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None


# Human-readable buckets for interpreting raw cProfile function names.
_BUCKET_RULES: list[tuple[str, Callable[[str], bool]]] = [
    ("oCSE driver", lambda n: "optimal_causation_entropy" in n),
    ("log-det / correlation minors", lambda n: "log_det" in n or "correlation_log" in n),
    ("discover_network (edge attrs)", lambda n: "discover_network" in n),
    ("shuffle_test / null CMI", lambda n: "shuffle_test" in n or "_null_cmi" in n),
    ("forward selection", lambda n: "forward" in n.lower() and "causationentropy" in n),
    ("backward elimination", lambda n: "backward" in n.lower() and "causationentropy" in n),
    ("Gaussian CMI batch", lambda n: "gaussian_conditional_mutual_information" in n),
    ("Poisson CMI batch", lambda n: "poisson_conditional_mutual_information" in n),
    ("conditional_mutual_information", lambda n: "conditional_mutual_information" in n),
    ("corrcoef / detcorr cache", lambda n: "corrcoef" in n or "detcorr" in n or "cached_" in n),
    ("Poisson entropy", lambda n: "poisson_" in n and "entropy" in n),
    ("k-NN entropy / neighbors", lambda n: "knn_entropy" in n or "tree_knn" in n or "neighbors_within" in n),
    ("k-NN / cdist", lambda n: "knn" in n.lower() or "cdist" in n or "KDTree" in n),
    ("geometric k-NN", lambda n: "geometric_knn" in n),
    ("KDE", lambda n: "kde" in n.lower()),
    ("LASSO", lambda n: "lasso" in n.lower()),
    ("numpy linalg", lambda n: "linalg" in n or "svd" in n or "det" in n),
]


def _synthetic_data(preset: str, seed: int, n: int, T: int):
    """Build benchmark data matching preset family (Gaussian vs Poisson)."""
    p = 0.2
    G_true = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    if preset.startswith("poisson"):
        data, _ = poisson_coupled_oscillators(n=n, T=T, seed=seed, G=G_true)
        return data, G_true
    data, _ = linear_stochastic_gaussian_process(
        rho=0.7, n=n, T=T, seed=seed, G=G_true
    )
    return data, G_true


def _bucket_name(func_name: str) -> str:
    for label, pred in _BUCKET_RULES:
        if pred(func_name):
            return label
    if "causationentropy" in func_name:
        return "other library"
    return "stdlib / third-party"


def _top_functions(stats: pstats.Stats, n: int) -> list[tuple[str, float]]:
    """Return top ``n`` causationentropy functions by exclusive time."""
    rows: list[tuple[str, float]] = []
    for (filename, line, func), (_cc, _nc, tottime, _ct, _callers) in stats.stats.items():
        if "causationentropy" not in filename:
            continue
        rows.append((f"{func} ({line})", tottime))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n]


def _aggregate_hotspots(stats: pstats.Stats, top_n: int) -> list[HotspotRow]:
    """Roll up exclusive (tottime) seconds into buckets — avoids double-counting nested calls."""
    bucket_totals: dict[str, float] = {}
    for (filename, line, func), (_cc, _nc, tottime, _ct, _callers) in stats.stats.items():
        qualified = f"{func} ({filename}:{line})"
        bucket = _bucket_name(qualified)
        bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + tottime

    total = sum(bucket_totals.values()) or 1.0
    ranked = sorted(bucket_totals.items(), key=lambda kv: kv[1], reverse=True)
    return [
        HotspotRow(
            label=label,
            cumulative_s=sec,
            share_pct=100.0 * sec / total,
            tottime_s=sec,
        )
        for label, sec in ranked[:top_n]
    ]


def _discover_from_preset(
    data: np.ndarray,
    preset: str,
    seed: int,
    n_jobs: int,
    n_shuffles: int,
) -> nx.MultiDiGraph:
    """Run discovery using preset hyperparameters with explicit call-site params."""
    cfg = get_discovery_preset(preset)
    k_neighbors = int(cfg.get("k", 5))
    return discover_network(
        data,
        method=str(cfg["method"]),
        information=str(cfg.get("information", "gaussian")),
        max_lag=int(cfg["max_lag"]),
        alpha_forward=float(cfg.get("alpha_forward", 0.05)),
        alpha_backward=float(cfg.get("alpha_backward", 0.05)),
        metric=str(cfg.get("metric", "euclidean")),
        bandwidth=str(cfg.get("bandwidth", "silverman")),
        k=k_neighbors,
        n_shuffles=n_shuffles,
        n_jobs=n_jobs,
        seed=seed,
        show_progress=False,
    )


def profile_preset(
    preset: str,
    seed: int,
    n_nodes: int,
    length: int,
    n_jobs: int,
    top_n: int,
    n_shuffles: int,
    top_functions_n: int,
) -> ProfileReport:
    """Time and profile ``discover_network`` for one preset."""
    get_discovery_preset(preset)  # validate name early
    data, _ = _synthetic_data(preset, seed, n_nodes, length)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    try:
        profiler.enable()
        G = _discover_from_preset(data, preset, seed, n_jobs, n_shuffles)
        profiler.disable()
        wall = time.perf_counter() - t0
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        hotspots = _aggregate_hotspots(stats, top_n)
        funcs = _top_functions(stats, top_functions_n) if top_functions_n > 0 else []
        return ProfileReport(
            preset=preset,
            wall_s=wall,
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            data_shape=data.shape,
            hotspots=hotspots,
            top_functions=funcs,
        )
    except Exception as exc:
        profiler.disable()
        return ProfileReport(
            preset=preset,
            wall_s=time.perf_counter() - t0,
            n_nodes=n_nodes,
            n_edges=0,
            data_shape=data.shape,
            error=f"{type(exc).__name__}: {exc}",
        )


def _default_presets() -> list[str]:
    names = list_presets()
    defaults = ["reproduction", "knn_standard", "poisson_standard"]
    return [p for p in defaults if p in names]


def _print_report(report: ProfileReport, show_detail: bool, show_functions: bool) -> None:
    print(f"\n{'=' * 72}")
    print(f"PRESET: {report.preset}")
    if report.error:
        print(f"  FAIL  {report.error}")
        return
    print(
        f"  wall={report.wall_s:.2f}s  shape={report.data_shape}  "
        f"nodes={report.n_nodes}  edges={report.n_edges}"
    )
    if not show_detail:
        return
    print(f"  {'bucket':32s} {'exclusive(s)':>12s} {'share':>7s}")
    print(f"  {'-' * 32} {'-' * 12} {'-' * 7}")
    for row in report.hotspots:
        print(
            f"  {row.label:32s} {row.tottime_s:12.3f} {row.share_pct:6.1f}%"
        )
    if show_functions and report.top_functions:
        print(f"  {'function':40s} {'exclusive(s)':>12s}")
        print(f"  {'-' * 40} {'-' * 12}")
        for name, sec in report.top_functions:
            print(f"  {name:40s} {sec:12.3f}")


def _resolve_n_shuffles(preset: str, tier: str, override: int | None) -> int:
    if override is not None:
        return override
    cfg = get_discovery_preset(preset)
    if tier == "full":
        return int(cfg["n_shuffles"])
    # Profile tier: cap shuffles and series length by estimator cost.
    if preset.startswith("knn") or preset.startswith("geometric"):
        return min(int(cfg["n_shuffles"]), 10)
    return min(int(cfg["n_shuffles"]), 80)


def _resolve_length(preset: str, tier: str, length: int) -> int:
    if tier == "full":
        return length
    if preset.startswith("knn") or preset.startswith("geometric"):
        return min(length, 100)
    return length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        action="append",
        help="Preset to profile (repeatable). Default: reproduction, knn_standard, poisson_standard.",
    )
    parser.add_argument(
        "--tier",
        choices=("profile", "full"),
        default="profile",
        help="profile caps n_shuffles at 80; full uses preset n_shuffles.",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=None,
        help="Override shuffle count for all presets (explicit, for A/B comparisons).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1, help="Use 1 for stable profiles.")
    parser.add_argument("--top", type=int, default=12, help="Hotspot buckets to show per preset.")
    parser.add_argument(
        "--top-functions",
        type=int,
        default=0,
        help="Also list top N causationentropy functions by exclusive time (0=off).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print wall times only (no bucket breakdown).",
    )
    args = parser.parse_args()

    presets = args.preset if args.preset else _default_presets()
    print("CausationEntropy profile hotspots", flush=True)
    print(
        f"  tier={args.tier}  n_jobs={args.n_jobs}  seed={args.seed}",
        flush=True,
    )

    reports: list[ProfileReport] = []
    for preset in presets:
        n_shuffles = _resolve_n_shuffles(preset, args.tier, args.n_shuffles)
        length = _resolve_length(preset, args.tier, args.length)
        print(
            f"  profiling {preset} (T={length}, n_shuffles={n_shuffles})...",
            flush=True,
        )
        reports.append(
            profile_preset(
                preset,
                args.seed,
                args.nodes,
                length,
                args.n_jobs,
                args.top,
                n_shuffles,
                args.top_functions,
            )
        )
        _print_report(
            reports[-1],
            show_detail=not args.summary_only,
            show_functions=args.top_functions > 0,
        )

    ok = [r for r in reports if r.error is None]
    if len(ok) > 1:
        print(f"\n{'=' * 72}")
        print("WALL-CLOCK SUMMARY")
        for r in sorted(ok, key=lambda x: x.wall_s, reverse=True):
            print(f"  {r.preset:24s} {r.wall_s:8.2f}s  edges={r.n_edges}")

    return 0 if all(r.error is None for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
