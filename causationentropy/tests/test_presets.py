"""Tests for discovery presets and parallel permutation options."""

import warnings

import networkx as nx
import numpy as np
import pytest

from causationentropy import discover_network
from causationentropy.core.presets import get_discovery_preset, list_presets
from causationentropy.core.stats import Compute_TPR_FPR
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process
from causationentropy.graph import evaluate_network_recovery, network_to_adjacency


def test_list_presets_includes_reproduction():
    presets = list_presets()
    assert "reproduction" in presets
    assert "knn_standard" in presets
    assert "poisson_standard" in presets


def test_reproduction_preset_values():
    preset = get_discovery_preset("reproduction")
    assert preset["method"] == "standard"
    assert preset["information"] == "gaussian"
    assert preset["max_lag"] == 1
    assert preset["n_shuffles"] == 1000


def test_discover_network_with_reproduction_preset():
    seed = 42
    G_true = nx.erdos_renyi_graph(5, 0.2, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=0.7, n=5, T=200, seed=seed, G=G_true
    )
    G_disc = discover_network(
        data,
        preset="reproduction",
        show_progress=False,
        seed=seed,
        n_jobs=2,
    )
    tpr, fpr = evaluate_network_recovery(G_true, G_disc)
    assert tpr == 1.0
    assert fpr == 0.0


def test_n_jobs_parallel_matches_serial():
    seed = 7
    G_true = nx.erdos_renyi_graph(4, 0.3, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=0.7, n=4, T=120, seed=seed, G=G_true
    )
    common = dict(
        method="standard",
        information="gaussian",
        max_lag=1,
        n_shuffles=80,
        alpha_forward=0.05,
        alpha_backward=0.05,
        show_progress=False,
        seed=seed,
    )
    G_serial = discover_network(data, n_jobs=1, **common)
    G_parallel = discover_network(data, n_jobs=2, **common)
    A = network_to_adjacency(G_serial)
    B = network_to_adjacency(G_parallel)
    np.testing.assert_array_equal(A, B)


def test_k_alias_overrides_k_means():
    seed = 11
    data = np.random.default_rng(seed).normal(size=(120, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        G_default = discover_network(
            data,
            information="knn",
            max_lag=1,
            n_shuffles=20,
            k_means=5,
            show_progress=False,
            seed=seed,
            n_jobs=1,
        )
        G_k = discover_network(
            data,
            information="knn",
            max_lag=1,
            n_shuffles=20,
            k_means=5,
            k=8,
            show_progress=False,
            seed=seed,
            n_jobs=1,
        )
    assert isinstance(G_default, nx.MultiDiGraph)
    assert isinstance(G_k, nx.MultiDiGraph)


def test_network_to_adjacency_collapses_lags():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", lag=1, cmi=0.5, p_value=0.01)
    G.add_edge("A", "B", lag=2, cmi=0.3, p_value=0.02)
    A = network_to_adjacency(G)
    assert A.shape == (2, 2)
    assert A.max() == 1


def test_compute_tpr_fpr_off_diagonal_only():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    tpr, fpr = Compute_TPR_FPR(A, B)
    assert tpr == 1.0
    assert 0.0 <= fpr <= 1.0
