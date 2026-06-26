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
    tree_neighbors_within_distance,
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


def test_gaussian_cmi_batch_matches_serial():
    """Batched forward-selection CMI must match one-column evaluations."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 5))
    Y = rng.normal(size=(n, 1))
    Z = rng.normal(size=(n, 2))

    from causationentropy.core.information.conditional_mutual_information import (
        gaussian_conditional_mutual_information,
        gaussian_conditional_mutual_information_batch,
    )

    batch = gaussian_conditional_mutual_information_batch(X, Y, Z)
    serial = np.array(
        [
            gaussian_conditional_mutual_information(X[:, [j]], Y, Z)
            for j in range(5)
        ]
    )
    np.testing.assert_allclose(batch, serial, rtol=1e-10, atol=1e-12)

    batch_marginal = gaussian_conditional_mutual_information_batch(X, Y, None)
    serial_marginal = np.array(
        [
            gaussian_conditional_mutual_information(X[:, [j]], Y, None)
            for j in range(5)
        ]
    )
    np.testing.assert_allclose(batch_marginal, serial_marginal, rtol=1e-10, atol=1e-12)


def test_tree_neighbors_within_distance_matches_dense():
    """Batched tree neighbor counts must match dense cdist thresholding."""
    rng = np.random.default_rng(6)
    data = rng.normal(size=(40, 3))
    js = rng.normal(size=(40, 2))
    epsilon = np.sort(cached_cdist(js))[:, 3]

    tree_counts = tree_neighbors_within_distance(data, epsilon, metric="euclidean")
    dense = cached_cdist(data, metric="euclidean")
    dense_counts = np.sum(dense < epsilon[:, None], axis=1) - 1

    np.testing.assert_allclose(tree_counts, dense_counts, rtol=0, atol=0)


def test_poisson_entropy_vectorized_regression():
    """Vectorized poisson_entropy matches fixed reference nats (regression guard)."""
    from causationentropy.core.information.entropy import poisson_entropy

    refs = {
        1.0: 1.3048422422562513,
        2.0: 1.7048826439329838,
        3.0: 1.9314701981485676,
    }
    for lam, expected in refs.items():
        got = poisson_entropy(lam)
        assert abs(got - expected) < 1e-10

    batch = poisson_entropy(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(batch, [refs[1.0], refs[2.0], refs[3.0]], rtol=1e-10)


def test_poisson_cmi_batch_matches_serial():
    """Batched Poisson forward-selection CMI must match one-column evaluations."""
    rng = np.random.default_rng(43)
    n = 80
    X = rng.poisson(3, size=(n, 4)).astype(float)
    Y = rng.poisson(3, size=(n, 1)).astype(float)
    Z = rng.poisson(3, size=(n, 2)).astype(float)

    from causationentropy.core.information.conditional_mutual_information import (
        poisson_conditional_mutual_information,
        poisson_conditional_mutual_information_batch,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        batch = poisson_conditional_mutual_information_batch(X, Y, Z)
        serial = np.array(
            [
                poisson_conditional_mutual_information(X[:, [j]], Y, Z)
                for j in range(4)
            ]
        )
        np.testing.assert_allclose(batch, serial, rtol=1e-10, atol=1e-12)

        batch_marginal = poisson_conditional_mutual_information_batch(X, Y, None)
        serial_marginal = np.array(
            [
                poisson_conditional_mutual_information(X[:, [j]], Y, None)
                for j in range(4)
            ]
        )
        np.testing.assert_allclose(batch_marginal, serial_marginal, rtol=1e-10, atol=1e-12)


def test_poisson_cmi_uses_corrcoef_cache():
    """Repeated Poisson CMI on the same block should hit the corrcoef cache."""
    rng = np.random.default_rng(4)
    X = rng.poisson(2, size=(60, 1)).astype(float)
    Y = rng.poisson(2, size=(60, 1)).astype(float)
    Z = rng.poisson(2, size=(60, 2)).astype(float)

    from causationentropy.core.information.conditional_mutual_information import (
        poisson_conditional_mutual_information,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        poisson_conditional_mutual_information(X, Y, Z)
        poisson_conditional_mutual_information(X, Y, Z)
    assert get_cache_stats()["corrcoef_cache_size"] == 1


def test_information_lasso_can_differ_from_lasso():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 8))
    Y = X[:, [0]] + 0.1 * rng.normal(size=(120, 1))
    lasso = lasso_optimal_causation_entropy(X, Y, rng)
    info = information_lasso_optimal_causation_entropy(X, Y, rng, n_shuffles=0)
    assert isinstance(lasso, list)
    assert isinstance(info, list)


def test_information_lasso_significance_prunes_lasso_survivors():
    """Permutation pruning must never add indices beyond the LASSO support."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(150, 12))
    Y = X[:, [0]] + 0.05 * rng.normal(size=(150, 1))

    lasso_screen = information_lasso_optimal_causation_entropy(
        X, Y, rng, n_shuffles=0
    )
    pruned = information_lasso_optimal_causation_entropy(
        X,
        Y,
        rng,
        n_shuffles=60,
        alpha=0.05,
        n_jobs=1,
    )
    assert set(pruned).issubset(set(lasso_screen))


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
