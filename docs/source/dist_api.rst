.. currentmodule:: attribench

Distributed API Reference
=========================
This page contains the reference for the distributed API of :mod:`attribench`.
For simple, small-scale experiments, the :mod:`attribench`
:doc:`functional API<func_api>` is recommended. For large-scale experiments,
the distributed API is recommended, as it can leverage multiple GPUs.

General
-------
.. autosummary::
    :toctree: generated

    attribench.distributed.SelectSamples
    attribench.distributed.TrainAdversarialPatches
    attribench.distributed.ComputeAttributions

Metrics
-------
.. autosummary::
    :toctree: generated

    attribench.distributed.Metric
    attribench.distributed.metrics.Deletion
    attribench.distributed.metrics.Insertion
    attribench.distributed.metrics.ImpactCoverage
    attribench.distributed.metrics.Irof
    attribench.distributed.metrics.Infidelity
    attribench.distributed.metrics.MaxSensitivity
    attribench.distributed.metrics.MinimalSubset
    attribench.distributed.metrics.SensitivityN
