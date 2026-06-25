#!/usr/bin/env python3
"""
Integration test for all entropy methods in causal discovery.

Each test delegates to :mod:`causationentropy.core.integration_benchmarks`
so pytest names stay aligned with named discovery presets.
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
    assert_integration_benchmark("test_standard_gaussian")


def test_alternative_gaussian():
    assert_integration_benchmark("test_alternative_gaussian")


def test_standard_knn():
    assert_integration_benchmark("test_standard_knn")


def test_alternative_knn():
    assert_integration_benchmark("test_alternative_knn")


def test_minkowski_standard_knn():
    assert_integration_benchmark("test_minkowski_standard_knn")


def test_standard_geometric_knn():
    assert_integration_benchmark("test_standard_geometric_knn")


def test_standard_kde():
    assert_integration_benchmark("test_standard_kde")


def test_information_lasso():
    assert_integration_benchmark("test_information_lasso")


def test_lasso():
    assert_integration_benchmark("test_lasso")


def test_standard_poisson():
    assert_integration_benchmark("test_standard_poisson")


def test_alternative_poisson():
    assert_integration_benchmark("test_alternative_poisson")


def test_alternative_geometric_knn():
    assert_integration_benchmark("test_alternative_geometric_knn")


def test_alternative_kde():
    assert_integration_benchmark("test_alternative_kde")


def test_kde_scott_bandwidth():
    assert_integration_benchmark("test_kde_scott_bandwidth")


def test_knn_chebyshev_metric():
    assert_integration_benchmark("test_knn_chebyshev_metric")


def test_knn_manhattan_metric():
    assert_integration_benchmark("test_knn_manhattan_metric")


def test_parameter_variations():
    """Ad-hoc parameter sweep (not tied to a single preset)."""
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

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Higher max_lag Gaussian Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.9
    assert fpr <= 0.1

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
