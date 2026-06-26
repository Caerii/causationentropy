# Changelog

All notable changes to this project are documented in this file.

## [1.2.10] - 2026-06-25

### Changed

- **k-NN neighbor counts** — ``tree_neighbors_within_distance`` uses one batched
  ``query_radius(..., count_only=True)`` call with per-sample radii instead of
  per-row sklearn queries (~19× faster on ``knn_standard`` profile tier).
- **Estimator caches** — thread-safe LRU eviction for parallel ``n_jobs`` shuffle
  tests (fixes ``KeyError`` under concurrent tree-cache access).
- **`poisson_entropy`** — vectorized Poisson PMF accumulation (array ``stack``
  instead of ``np.matrix`` list); ~2.5× faster on ``poisson_standard`` profile tier.

### Added

- **`test_tree_neighbors_within_distance_matches_dense`** — tree vs dense parity.
- **Poisson entropy regression guard** in ``test_distance_cache``.

### Notes (profile tier, ``n_jobs=1``, seed=42)

- ``knn_standard``: wall ~247s → ~11s (``T=100``, 10 shuffles).
- ``poisson_standard``: wall ~77s → ~30s (30 shuffles).
- ``reproduction`` (Gaussian): unchanged (~2.2s).

## [1.2.9] - 2026-06-25

### Added

- **`examples/profile_hotspots.py`** — empirical profiler with ``--tier profile|full``,
  preset-aware ``n_shuffles``/``T`` caps, exclusive-time bucket summaries, and
  optional ``--top-functions`` drill-down.

### Notes (profile tier, ``n_jobs=1``, seed=42)

Initial hotspot study on local hardware:

- **Gaussian ``reproduction``** (~2s): log-det / correlation minors and
  ``gaussian_conditional_mutual_information_batch`` dominate; backward elimination,
  edge re-labeling, and shuffle tests are negligible at this tier.
- **k-NN ``knn_standard``** (~200s at ``T=100``, 10 shuffles): bottleneck is
  ``tree_neighbors_within_distance`` plus sklearn array validation — not forward
  batching or backward pruning.
- **Poisson ``poisson_standard``** (~77s at 30 shuffles): ``poisson_entropy`` /
  scipy ``pmf`` dominate; ``poisson_conditional_mutual_information_batch`` is already cheap.

## [1.2.8] - 2026-06-25

### Changed

- **CONTRIBUTING compliance** — removed ``*args``/``**kwargs`` signatures and dict
  unpacking at ``discover_network`` call sites in ``integration_benchmarks``,
  ``explore_profile``, and tests; clarified call-site rule in CONTRIBUTING.md.

## [1.2.7] - 2026-06-25

### Added

- **`examples/explore_profile.py`** — exploration harness with argparse for the
  integration benchmark matrix, reproduction grids, scaling sweeps, edge cases,
  and optional ``cProfile`` snapshots (``--list``, ``--matrix``, ``--quick``,
  ``--all``).

## [1.2.6] - 2026-06-25

### Added

- **`poisson_conditional_mutual_information_batch`** — evaluates Poisson CMI for all
  remaining forward-selection candidates from one ``corrcoef`` pass (principal
  submatrix per candidate).

### Changed

- **Forward selection** — ``alternative_forward`` and ``standard_forward`` call
  the batched Poisson path when ``information='poisson'``.

## [1.2.5] - 2026-06-25

### Changed

- **`information_lasso`** — after MI-weighted LASSO screening, survivors are pruned with
  :func:`backward` permutation tests (``n_shuffles``, ``alpha_backward`` from
  :func:`discover_network` / presets). Set ``n_shuffles=0`` to skip pruning.
- **`discover_network`** — forwards ``alpha_backward``, ``n_shuffles``, ``information``,
  ``metric``, ``k_neighbors``, ``bandwidth``, and ``n_jobs`` explicitly to
  ``information_lasso_optimal_causation_entropy`` (no ``**kwargs`` plumbing).

## [1.2.4] - 2026-06-25

### Added

- **`gaussian_conditional_mutual_information_batch`** — evaluates CMI for all
  remaining forward-selection candidates from one ``corrcoef`` pass.

### Changed

- **Forward selection** — ``standard_forward`` and ``alternative_forward`` call
  the batched Gaussian path when ``information='gaussian'``.
- **Poisson CMI** — joint correlation matrices use ``cached_corrcoef`` so
  repeated evaluations on the same data block avoid redundant work.

## [1.2.3] - 2026-06-25

### Changed

- **Documentation pass** — expanded module and function docstrings across
  ``integration_benchmarks``, ``distance_cache``, ``entropy``, Gaussian MI/CMI,
  ``presets``, and ``discovery`` with explanatory prose and preserved inline
  comments that walk through algorithms step by step.
- **Integration tests** — each pytest function now documents which preset it
  validates; ``test_parameter_variations`` retains its original illustrative
  comments.

## [1.2.2] - 2026-06-25

### Added

- **`integration_benchmarks` module** — maps each `test_data_integration.py` test to a named preset with shared `run_integration_benchmark()` / `assert_integration_benchmark()` helpers.
- **API docs** — presets and integration benchmark functions in `network_discovery.rst`.

### Changed

- **Integration tests** — refactored to run via preset-backed benchmark matrix (473 → ~150 lines).
- **CONTRIBUTING** — integration test ↔ preset table and local one-liner for single benchmarks.
- **CI** — integration job comment pointing at the benchmark matrix.
- **Geometric k-NN entropy** — batched hyperellipsoid checks and vectorized singular-value ratio term.

## [1.2.1] - 2026-06-25

### Added

- **`examples/benchmark_matrix.py`** — CLI timing matrix over discovery presets with TPR/FPR reporting (`--quick`, `--all`).

### Changed

- **README** — documents uv dev workflow, presets, performance knobs (`n_jobs`, `use_cache`), and the distinction between interactive defaults and `reproduction` settings.
- **`discover_network` docstring** — explains default parameters vs validated presets.
- **Gaussian MI/CMI** — single `cached_corrcoef` call per evaluation with submatrix log-determinants (fewer redundant `corrcoef` passes).
- **Geometric k-NN entropy** — vectorized neighbor indexing and distance accumulation.
- **Parallel shuffle tests** — `functools.partial` binding and tuned `chunksize` for `ThreadPoolExecutor.map`.

## [1.2.0] - 2026-06-25

### Added

- **Discovery presets** — 18 named configurations from integration benchmarks (`reproduction`, `knn_standard`, `poisson_standard`, `logistic_chaos`, `demo`, etc.) via `discover_network(..., preset=...)`, with `list_presets()`, `get_discovery_preset()`, and `describe_preset()`.
- **Parallel permutation tests** — `n_jobs` on `discover_network` and `shuffle_test` now parallelizes shuffle CMI evaluation with `ThreadPoolExecutor`.
- **Progress and logging controls** — `show_progress` (tqdm over target variables), `verbose` (per-node status lines), and `seed` for reproducible permutations.
- **`k` parameter** — clearer alias for k-NN neighbor count (supersedes the misleading `k_means` name in docs).
- **Evaluation helpers** — `network_to_adjacency()` and `evaluate_network_recovery()` in `causationentropy.graph`.
- **Reproduction tooling** — `examples/reproduce_benchmark.py` CLI for running validated presets locally.
- **Distance and correlation caches** — new `causationentropy.core.information.distance_cache` module with `cached_cdist`, `cached_detcorr`, spatial-tree k-NN queries, `configure_cache_for_discovery()`, and `use_cache` on `discover_network`.
- **Information-theoretic LASSO** — `information_lasso` method now scales predictors by Gaussian MI with the target before LASSO selection (distinct from plain `lasso`).
- **Logistic dynamics benchmark** — `logistic_chaos` preset and integration test for nonlinear coupled logistic-map data.
- **CI integration job** — GitHub Actions workflow runs `test_data_integration.py` on Ubuntu (Python 3.11).
- **uv development config** — `[tool.uv]`, `[dependency-groups]`, and `uv.lock` for reproducible dev environments.

### Changed

- **Matplotlib 3.11+ compatibility** — `plot_causal_network` uses `mpl.colormaps` with a legacy fallback instead of removed `plt.cm.get_cmap`.
- **TPR/FPR evaluation** — `Compute_TPR_FPR` now evaluates off-diagonal entries only so FPR stays in `[0, 1]`.
- **Geometric k-NN CMI** — passes through `k` when `Z=None` (marginal case matches requested neighbor count).
- **Package exports** — top-level `__version__` aligned with packaging metadata; `tests` removed from public `causationentropy` exports.
- **Documentation** — README and CONTRIBUTING updated for presets, uv setup, and integration-test reproduction workflow.

### Fixed

- Plotting tests failing on matplotlib ≥ 3.11 (22 `plot_causal_network` tests).
- Documented but unimplemented `n_jobs` on causal discovery.
- Stale `__version__ = "0.1.0"` in `causationentropy/__init__.py` (now matches release).

## [1.1.0] - 2025-11-12

### Added

- Support for converting causal networks to pandas DataFrames.
- Utility for importing and processing Tigramite network structures.
- Companion matrix calculation for better FPR/TPR calculations.
- `plot_causal_network` with automated circular layout algorithms.
- Non-negative conditional mutual information and mutual information clamping.

## [1.0.0] - 2025-10-01

### Added

- Full test coverage.
- Full API design.

## [0.1.0] - 2025-09-15

### Added

- All math code parsed to new API.
