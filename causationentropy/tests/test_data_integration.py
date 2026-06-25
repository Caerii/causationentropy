#!/usr/bin/env python3
"""
Integration tests for causal discovery across all supported information estimators.

These tests are the reproduction gate for the library: each one generates
synthetic data whose distributional assumptions match a particular information
estimator, runs :func:`~causationentropy.discover_network` with a named preset,
and checks that the recovered graph achieves the true-positive and false-positive
rates recorded in :mod:`causationentropy.core.integration_benchmarks`.

The test bodies are intentionally thin. All shared logic (data generation,
preset lookup, TPR/FPR evaluation) lives in the benchmark matrix so this file
reads as a catalog of *what* we validate, not *how* the plumbing works.
"""

import warnings

import networkx as nx
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from causationentropy import discover_network
from causationentropy.core.integration_benchmarks import assert_integration_benchmark
from causationentropy.core.stats import Compute_TPR_FPR
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process

pytestmark = pytest.mark.integration


def test_standard_gaussian():
    """Standard oCSE + Gaussian CMI on linear VAR data (preset: reproduction)."""
    assert_integration_benchmark("test_standard_gaussian")


def test_alternative_gaussian():
    """Alternative oCSE + Gaussian CMI (preset: gaussian_alternative)."""
    assert_integration_benchmark("test_alternative_gaussian")


def test_standard_knn():
    """Standard oCSE + k-NN CMI, Euclidean metric (preset: knn_standard)."""
    assert_integration_benchmark("test_standard_knn")


def test_alternative_knn():
    """Alternative oCSE + k-NN with high k and many shuffles (preset: knn_alternative)."""
    assert_integration_benchmark("test_alternative_knn")


def test_minkowski_standard_knn():
    """Standard k-NN CMI with Minkowski metric (preset: knn_minkowski)."""
    assert_integration_benchmark("test_minkowski_standard_knn")


def test_standard_geometric_knn():
    """Standard oCSE + geometric k-NN CMI (preset: geometric_knn_standard)."""
    assert_integration_benchmark("test_standard_geometric_knn")


def test_standard_kde():
    """Standard oCSE + KDE CMI, Silverman bandwidth (preset: kde_standard)."""
    assert_integration_benchmark("test_standard_kde")


def test_information_lasso():
    """MI-weighted LASSO screening baseline (preset: information_lasso)."""
    assert_integration_benchmark("test_information_lasso")


def test_lasso():
    """Pure LASSO baseline without permutation selection (preset: lasso)."""
    assert_integration_benchmark("test_lasso")


def test_standard_poisson():
    """Standard oCSE + Poisson CMI on count data (preset: poisson_standard)."""
    assert_integration_benchmark("test_standard_poisson")


def test_alternative_poisson():
    """Alternative oCSE + Poisson CMI (preset: poisson_alternative)."""
    assert_integration_benchmark("test_alternative_poisson")


def test_alternative_geometric_knn():
    """Alternative oCSE + geometric k-NN (preset: geometric_knn_alternative)."""
    assert_integration_benchmark("test_alternative_geometric_knn")


def test_alternative_kde():
    """Alternative oCSE + KDE CMI (preset: kde_alternative)."""
    assert_integration_benchmark("test_alternative_kde")


def test_kde_scott_bandwidth():
    """KDE CMI with Scott bandwidth (preset: kde_scott)."""
    assert_integration_benchmark("test_kde_scott_bandwidth")


def test_knn_chebyshev_metric():
    """k-NN CMI with Chebyshev metric (preset: knn_chebyshev)."""
    assert_integration_benchmark("test_knn_chebyshev_metric")


def test_knn_manhattan_metric():
    """k-NN CMI with Manhattan (cityblock) metric (preset: knn_manhattan)."""
    assert_integration_benchmark("test_knn_manhattan_metric")


def test_parameter_variations():
    """Ad-hoc parameter sweep — not tied to a single named preset.

    This test explores sensitivity to ``max_lag`` and ``k_means`` on a smaller
    graph. It remains inline (rather than matrix-backed) because it exercises
    two custom configurations in one function rather than one validated preset.
    """
    T = 150
    rho = 0.8
    n_nodes = 4
    seed = 123
    p = 0.3
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )

    # Higher max_lag: more predictors enter the conditioning sets.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="gaussian",
        max_lag=3,
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=500,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Higher max_lag Gaussian Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.9
    assert fpr <= 0.1

    # Higher k for k-NN: smoother local density estimates, often more conservative.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="knn",
        metric="euclidean",
        max_lag=2,
        k_means=15,
        alpha_forward=0.001,
        alpha_backward=0.001,
        n_shuffles=800,
    )
    B = nx.to_numpy_array(G_discovered)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Higher k_means KNN Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.6
    assert fpr <= 0.1


if __name__ == "__main__":
    test_standard_gaussian()
    test_alternative_gaussian()
    test_standard_knn()
    test_alternative_knn()
    test_minkowski_standard_knn()
    test_standard_geometric_knn()
    test_standard_kde()
    test_information_lasso()
    test_lasso()
    test_standard_poisson()
    test_alternative_poisson()
    test_alternative_geometric_knn()
    test_alternative_kde()
    test_kde_scott_bandwidth()
    test_knn_chebyshev_metric()
    test_knn_manhattan_metric()
    test_parameter_variations()
