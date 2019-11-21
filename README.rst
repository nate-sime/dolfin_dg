************************
dolfin_dg utility module
************************


What does it do?
================

dolfin_dg provides utility function for the automatic generation of nonlinear DG FEM formulations.


Specifically, what does it do?
------------------------------

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


More details and citing
-----------------------

Paul Houston and Nathan Sime, 
*Automatic symbolic computation for discontinuous Galerkin finite element methods*,
SIAM Journal on Scientific Computing, 2018, 40(3), C327–C357, https://doi.org/10.1137/17M1129751.

Preprint https://arxiv.org/abs/1804.02338.


Dependencies
============

dolfin_dg depends on the core components of the FEniCS project (https://fenicsproject.org/).

dolfin_dg requires python 3.


Automated testing
=================

.. image:: https://img.shields.io/bitbucket/pipelines/nate-sime/dolfin_dg
   :target: https://bitbucket.org/nate-sime/dolfin_dg/addon/pipelines/home
   :alt: Pipelines Build Status

Unit tests are provided in ``test/unit/test_*.py``.


Installation
============

Docker image
------------


.. image:: https://quay.io/repository/natesime/dolfin_dg/status


Follow the instructions for installing https://fenicsproject.org/download/. A docker image 
of the dolfin_dg master branch is available:


.. code-block:: bash

    docker run -it quay.io/natesime/dolfin_dg:master

Custom installation
-------------------

Install in the usual way with your preferred prefix:

.. code-block:: bash
     
    python3 setup.py install --prefix=$HOME/local


Installation inside existing docker container
---------------------------------------------

Follow the instructions to install FEniCS with docker https://fenicsproject.org/download/.

Run docker and clone dolfin_dg

.. code-block:: bash

    docker run -it quay.io/fenicsproject/dev
    git clone https://bitbucket.org/nate-sime/dolfin_dg.git

Install dolfin_dg

.. code-block:: bash

    cd dolfin_dg
    sudo python3 setup.py install


Contributors
============

* Nate J. C. Sime nsime@carnegiescience.edu
* Paul Houston Paul.Houston@nottingham.ac.uk


License
=======

GNU LGPL, version 3.