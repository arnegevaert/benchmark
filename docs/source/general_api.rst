.. currentmodule:: attribench

General API Reference
=====================
This page contains the API reference for the attribench package.

Computations and Metrics
------------------------
Computations and metrics are divided into two API groups:
a :doc:`functional API<func_api>` and an
:doc:`object-oriented API<dist_api>`.
The functional API is the simplest to use, but
**does not support multi-GPU computations.**
If you want to run a large-scale
benchmarking experiment on multiple GPUs,
you should use the object-oriented API.

.. toctree::
    :maxdepth: 1

    func_api
    dist_api

Results
-------
.. autosummary::
    :toctree: generated

    attribench.result.MetricResult
    attribench.result.GroupedMetricResult
    attribench.result.DeletionResult
    attribench.result.InsertionResult
    attribench.result.ImpactCoverageResult
    attribench.result.InfidelityResult
    attribench.result.MaxSensitivityResult
    attribench.result.MinimalSubsetResult
    attribench.result.SensitivityNResult

Data
----
.. autosummary::
    :toctree: generated

    attribench.data.IndexDataset
    attribench.data.AttributionsDataset
    attribench.data.HDF5Dataset

Masking
-------
.. autosummary::
    :toctree: generated

    attribench.masking.Masker
    attribench.masking.ConstantMasker
    attribench.masking.RandomMasker
    attribench.masking.SampleAverageMasker
    attribench.masking.TabularMasker
    attribench.masking.ImageMasker
    attribench.masking.BlurringMasker

Plot
----
.. autosummary::
    :toctree: generated

    attribench.plot.ClusterPlot

Other
-----
.. autosummary::
    :toctree: generated

    attribench.ModelFactory
    attribench.MethodFactory
    attribench.AttributionMethod