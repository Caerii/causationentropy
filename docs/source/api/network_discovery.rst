Network Discovery
=================

Core algorithms for causal network discovery from time series data.

Discovery Presets
-----------------

Named presets encode integration-test validated settings. Use ``preset="reproduction"``
for paper-style benchmarks or ``preset="demo"`` for fast interactive runs.

.. autofunction:: causationentropy.core.presets.list_presets

.. autofunction:: causationentropy.core.presets.get_discovery_preset

.. autofunction:: causationentropy.core.presets.describe_preset

Integration Benchmarks
----------------------

Each integration test in ``test_data_integration.py`` maps to a preset via
``causationentropy.core.integration_benchmarks``.

.. autofunction:: causationentropy.core.integration_benchmarks.list_integration_benchmarks

.. autofunction:: causationentropy.core.integration_benchmarks.preset_for_integration_test

.. autofunction:: causationentropy.core.integration_benchmarks.run_integration_benchmark

Main Discovery Function
-----------------------

.. autofunction:: causationentropy.core.discovery.discover_network

Method Implementations
----------------------

.. autofunction:: causationentropy.core.discovery.standard_optimal_causation_entropy

.. autofunction:: causationentropy.core.discovery.alternative_optimal_causation_entropy

.. autofunction:: causationentropy.core.discovery.information_lasso_optimal_causation_entropy

.. autofunction:: causationentropy.core.discovery.lasso_optimal_causation_entropy

Selection Algorithms
--------------------

.. autofunction:: causationentropy.core.discovery.standard_forward

.. autofunction:: causationentropy.core.discovery.alternative_forward

.. autofunction:: causationentropy.core.discovery.backward

Statistical Testing
-------------------

.. autofunction:: causationentropy.core.discovery.shuffle_test

Linear Algebra Utilities
------------------------

.. autofunction:: causationentropy.core.linalg.correlation_log_determinant

Statistical Utilities
---------------------

.. autofunction:: causationentropy.core.stats.auc

.. autofunction:: causationentropy.core.stats.Compute_TPR_FPR

Visualization
-------------

.. autofunction:: causationentropy.core.plotting.roc_curve