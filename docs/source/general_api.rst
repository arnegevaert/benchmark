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
    :template: custom-class-template.rst

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
    :template: custom-class-template.rst

    attribench.data.IndexDataset
    attribench.data.AttributionsDataset
    attribench.data.HDF5Dataset

Masking
-------
.. autosummary::
    :toctree: generated
    :template: custom-class-template.rst

    attribench.masking.Masker
    attribench.masking.image.ImageMasker
    attribench.masking.image.BlurringImageMasker
    attribench.masking.image.ConstantImageMasker
    attribench.masking.image.RandomImageMasker
    attribench.masking.image.SampleAverageImageMasker
    attribench.masking.TabularMasker

Plot
----
.. autosummary::
    :toctree: generated
    :template: custom-class-template.rst

    attribench.plot.Plot
    attribench.plot.WilcoxonSummaryPlot
    attribench.plot.InterMetricCorrelationPlot
    attribench.plot.InterMethodCorrelationPlot
    attribench.plot.ConvergencePlot
    attribench.plot.KrippendorffAlphaPlot
    attribench.plot.ClusterPlot
    attribench.plot.MADRatioPlot
    attribench.plot.WilcoxonBarPlot

Other
-----
.. autosummary::
    :toctree: generated
    :template: custom-class-template.rst

    attribench.ModelFactory
    attribench.BasicModelFactory
    attribench.MethodFactory
    attribench.AttributionMethod