dolfin_dg utility module
========================

What does it do?
================

dolfin_dg provides utility function for the automatic generation of nonlinear DG FEM formulations.

Specifically, what does it do?
==============================

Consider the nonlinear conservation equation

.. math::

    \nabla \cdot \mathcal{L}(u; \nabla u) = f 

Here :math:`\mathcal{L}(\cdot; \cdot)` is a semilinear operator (nonlinear in the first argument and linear in the second). The semilinear residual weak formulation, subject to appropriate boundary conditions, reads: find :math:`u \in V` such that

.. math::

    \mathcal{N}(u; v) := 
    \int_D \mathcal{L}(u; \nabla u) : \nabla v \; \mathrm{d}x -
    \int_{\partial D} \mathcal{L}(u; \nabla u) \cdot n \cdot v \; \mathrm{d} s -
    \int_D f \cdot v \; \mathrm{d} x \equiv 0 \quad \forall v \in V.

The DG FEM formulation of the above equation is notoriously verbose to define. Programming the code to compute the DG FEM approximation :math:`u_h \in V_h` is a large task. dolfin_dg provides utility functions for the automatic formulation of the DG FEM discretisation of the weak formulation. These functions work with DOLFIN to facilitate simple implementation and efficient computation of the DG approximation of nonlinear FEM problems.

Dependencies
============

dolfin_dg depends on the core components of the FEniCS project (https://fenicsproject.org/).


Installation
============

Install in the usual way with your preferred prefix:

.. code-block:: bash
     
    python setup.py install --prefix=$HOME/local


Installation with docker
========================

Follow the instructions to install FEniCS with docker https://fenicsproject.org/download/.

Run docker and clone dolfin_dg

.. code-block:: bash

    docker run -it quay.io/fenicsproject/dev
    git clone https://bitbucket.org/nate-sime/dolfin_dg.git

Ensure you export the local PYTHONPATH

.. code-block:: bash

    export PYTHONPATH=$HOME/local/lib/python2.7/site-packages:$PYTHONPATH

Install dolfin_dg

.. code-block:: bash

    cd dolfin_dg
    python setup.py install --prefix=$HOME/local


Contributors
============

* Nate J. C. Sime njcs4@cam.ac.uk
* Paul Houston Paul.Houston@nottingham.ac.uk

License
=======

GNU LGPL, version 3.