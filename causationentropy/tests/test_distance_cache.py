"""Tests for distance caching and information-lasso weighting."""

import warnings

import networkx as nx
import numpy as np
import pytest

from causationentropy import discover_network
from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
    lasso_optimal_causation_entropy,
)
from causationentropy.core.information.distance_cache import (
    cached_cdist,
    cached_detcorr,
    clear_caches,
    configure_cache_for_discovery,
    get_cache_stats,
)
from causationentropy.core.information.mutual_information import knn_mutual_information
from causationentropy.datasets.synthetic import logisic_dynamics
from causationentropy.graph import evaluate_network_recovery


@pytest.fixture(autouse=True)
def reset_caches():
    clear_caches()
    yield
    clear_caches()


def test_cached_detcorr_reuses_result():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    first = cached_detcorr(data)
    second = cached_detcorr(data)
    assert first == second
    assert get_cache_stats()["detcorr_cache_size"] == 1


def test_cached_cdist_reuses_result():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(20, 2))
    first = cached_cdist(data)
    second = cached_cdist(data)
    np.testing.assert_array_equal(first, second)
    assert get_cache_stats()["distance_cache_size"] == 1


def test_configure_cache_for_discovery():
    config = configure_cache_for_discovery(
        (200, 5), max_lag=2, information_method="knn"
    )
    assert config["cache_size"] >= 16
    assert "estimated_memory_mb" in config


def test_knn_tree_and_dense_agree():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    Y = rng.normal(size=(80, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dense = knn_mutual_information(X, Y, metric="euclidean", k=3, kd_tree=False)
        tree = knn_mutual_information(X, Y, metric="euclidean", k=3, kd_tree=True)
    assert np.isfinite(dense)
    assert np.isfinite(tree)
    assert abs(dense - tree) < 0.5


def test_information_lasso_can_differ_from_lasso():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 8))
    Y = X[:, [0]] + 0.1 * rng.normal(size=(120, 1))
    lasso = lasso_optimal_causation_entropy(X, Y, rng)
    info = information_lasso_optimal_causation_entropy(X, Y, rng)
    assert isinstance(lasso, list)
    assert isinstance(info, list)


@pytest.mark.integration
def test_logistic_dynamics_end_to_end():
    data, adjacency = logisic_dynamics(n=5, t=400, seed=42)
    G_true = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    G_disc = discover_network(
        data,
        preset="logistic_chaos",
        show_progress=False,
        seed=42,
        n_jobs=2,
    )
    assert G_disc.number_of_nodes() == 5
    assert isinstance(G_disc, nx.MultiDiGraph)
    tpr, fpr = evaluate_network_recovery(G_true, G_disc)
    assert 0.0 <= tpr <= 1.0
    assert 0.0 <= fpr <= 1.0
