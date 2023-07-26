AttriBench: Metrics for Feature Attribution Techniques
======================================================
**AttriBench** is a `Pytorch <https://pytorch.org/>`_-based implementation of
several metrics for the evaluation of feature attribution maps and methods.
AttriBench provides a functional and an object-oriented API for the computation
of these metrics, along with a set of utility functions for the necessary
preparations (e.g. computing attribution maps) as well as for the visualization
of the results.

The **functional API** is generally easier to use, and can be used to get
started quickly if the scale of the evaluation is not too large. The
**object-oriented API** is more flexible and can use multiple GPUs for
evaluation of large datasets.

For more information, see the `documentation <https://attribench.readthedocs.io/>`_.

Installation
------------
AttriBench can be installed from PyPI using pip:

.. code-block:: bash
    
    pip install attribench
