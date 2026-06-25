# Changelog

All notable changes to this project are documented in this file.

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
