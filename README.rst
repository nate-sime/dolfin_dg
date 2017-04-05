`dolfin_dg` utility module
========================

Install in the usual way with your preferred prefix:

.. code-block:: bash
     
    python setup.py install --prefix=$HOME/local


What does it do?
================

Consider the partial differential equation

.. math::

    \mathcal{L}(u; \nabla u) = f

where :math:`\mathcal{L}(\cdot; \cdot) : V \rightarrow \mathbb{R}` is a semilinear differential operator. Consider example of a standard hyperbolic
equation

.. math::

    \nabla \cdot \mathcal{F}^c(u) = f

where :math:`\mathcal{F}^c(\cdot) : V \rightarrow \mathbb{R}`. 

Why do we do this?
==================

DG formulations are horribly complicated. This project aims to reduce human
error by automating their generation.