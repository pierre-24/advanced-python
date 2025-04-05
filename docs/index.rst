``neighbour_search`` documentation
==================================

This package implements different approaches to perform a ball search and a kNN search, in 2D.

.. contents:: On this page
   :local:
   :backlinks: none

Installation
------------

To use the ``neighbour_search`` package:

.. code-block:: bash

   pip install git+https://github.com/pierre-24/advanced-python.git

To follow the lecture:

.. code-block:: bash

   git clone https://github.com/pierre-24/advanced-python.git
   cd advanced-python
   # optional: use a virtualenv
   python -m venv venv
   source venv/bin/activate
   # install dev packages
   pip install -e.[dev]

If you want to contribute, follow the previous instructions, but use a fork instead.

Example
-------

Two examples are provided.
With a working matplolib installation (with a GUI frontend), you can use:

.. code-block:: bash

   # use ball_search()
   python ./neighbour_search/scripts/example_ball_search.py
   # use knn_search()
   python ./neighbour_search/scripts/example_knn_search.py

They both display a set of points, and highlight the one found, depending on the script.
You can switch between different implementations using ``-t``.
Use ``--help`` for other options.

Performances
------------

.. figure:: ./performances.svg
   :width: 600px
   :align: center
   :alt: Performance test

   Timing for a ball search (:math:`\frac{\sqrt{N}}{10}` points) and kNN (20 neighbour) selection among :math:`N` points, randomly positioned in a :math:`\sqrt{N}\times\sqrt{N}` box.
   Leaf size is 100.
   Each selection is repeated 25 times and a mean and standard deviation is then computed.
   Performance test performed on AMD Ryzen 7 7700.
   The error bars indicate :math:`[-2\sigma,2\sigma]`, while the dashed lines indicate the (supposed) computational complexity.
   Notice the logarithmic scale.

API
---

Base
++++

.. automodule:: neighbour_search
   :members:
   :show-inheritance:
   :undoc-members:

Naive implementation
++++++++++++++++++++

.. automodule:: neighbour_search.naive_ns
   :members:
   :show-inheritance:
   :undoc-members:

k-d tree implementation
+++++++++++++++++++++++

.. automodule:: neighbour_search.kdtree_ns
   :members:
   :show-inheritance:
   :undoc-members:

Ball tree implementation
++++++++++++++++++++++++

.. automodule:: neighbour_search.balltree_ns
   :members:
   :show-inheritance:
   :undoc-members: