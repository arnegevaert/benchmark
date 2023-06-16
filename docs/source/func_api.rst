.. currentmodule:: attribench

Functional API Reference
========================
This page contains the reference for the functional API of :mod:`attribench`.
The functional API is generally easier to use for simple cases, but note
that the functional API **does not support multi-GPU computations.**

If you want to run a large-scale benchmarking experiment on multiple GPUs,
you should use the :mod:`attribench` :doc:`object-oriented API<dist_api>` instead.

General
-------
.. autosummary::
    :toctree: generated/

    attribench.functional.select_samples
    attribench.functional.train_adversarial_patches
    attribench.functional.compute_attributions

Metrics
-------
.. autosummary::
    :toctree: generated/

    attribench.functional.metrics.deletion
    attribench.functional.metrics.insertion
    attribench.functional.metrics.irof
    attribench.functional.metrics.impact_coverage