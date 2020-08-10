
Welcome to dolfin_dg's documentation!
=====================================

``dolfin_dg`` provides utility functions for the automatic generation of
nonlinear DG FEM formulations using `UFL <https://github.com/FEniCS/ufl>`_.

Consider the nonlinear conservation equation

.. math :: \nabla \cdot \mathcal{L}(u; \nabla u) = f

Here :math:`\mathcal{L}(\cdot; \cdot)` is a semilinear operator (nonlinear in
the first argument and linear in the second). The semilinear residual weak
formulation, subject to appropriate boundary conditions, reads: find
:math:`u \in V` such that

.. math ::

   \mathcal{N}(u; v) = \int_D \mathcal{L}(u; \nabla u) : \nabla v \; \mathrm{d}x
   - \int_{\partial D} \mathcal{L}(u; \nabla u) \cdot n \cdot v \; \mathrm{d}s
   - \int_D f \cdot v \; \mathrm{d}x \equiv 0
   \quad \forall v \in V.

The DG FEM formulation of the above equation is notoriously verbose to define.
Programming the code to compute the DG FEM approximation :math:`u_h \in V_h`
is a large task. ``dolfin_dg`` provides utility functions for the automatic
formulation of the DG FEM discretisation of the weak formulation. These
functions work with UFL to facilitate simple implementation and efficient
computation of the DG approximation of nonlinear FEM problems.

Contributors
~~~~~~~~~~~~
* `Nate Sime <nsime@carnegiescience.edu>`_ (Carnegie Institution for Science)
* `Paul Houston <Paul.Houston@nottingham.ac.uk>`_ (University of Nottingham)
* `Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>`_ (Oxford University)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

..  toctree::
    :maxdepth: 2
    :caption: Contents:

    tree/aero.rst
    tree/nitsche.rst
    tree/operators.rst
