# dolfin_dg

![unit tests badge](https://github.com/nate-sime/dolfin_dg/actions/workflows/run_tests.yml/badge.svg
)

`dolfin_dg` provides utility functions for the automatic generation of nonlinear
DG FEM formulations using [UFL](https://github.com/FEniCS/ufl).

`dolfin_dg` derives its name from original development with the DOLFIN
component of the FEniCS project. However, the core components of `dolfin_dg` now
solely depend on UFL. 

`dolfin_dg` has been successfully employed with:

* [DOLFINx](https://github.com/FEniCS/dolfinx)
* [DOLFIN](https://bitbucket.org/fenics-project/dolfin)
* [Firedrake](https://www.firedrakeproject.org/)
* [dune-fem](https://www.dune-project.org/modules/dune-fem/)

## Specifically, what does it do?

Consider the nonlinear conservation equation

$$
\nabla \cdot \mathcal{L}(u; \nabla u) = f
$$

Here $\mathcal{L}(\cdot; \cdot)$ is a semilinear operator (nonlinear in the first argument and linear in the
second). The semilinear residual weak formulation, subject to appropriate
boundary conditions, reads: find $u \in V$ such that

$$
\mathcal{N}(u; v) = \int_D \mathcal{L}(u; \nabla u) : \nabla v \; \mathrm{d}x - \int_{\partial D} \mathcal{L}(u; \nabla u) \cdot n \cdot v \; \mathrm{d} s - \int_D f \cdot v \; \mathrm{d} x \equiv 0 \quad \forall v \in V.
$$

The DG FEM formulation of the above equation is notoriously verbose to define.
Programming the code to compute the DG FEM approximation $u_h \in V_h$ is a large
task. `dolfin_dg` provides utility functions for the automatic formulation of
the DG FEM discretisation of the weak formulation. These functions work with
UFL to facilitate simple implementation and efficient computation of the DG
approximation of nonlinear FEM problems.


## More details and citing

Paul Houston and Nathan Sime,  
*Automatic symbolic computation for discontinuous Galerkin finite element methods*,  
[SIAM Journal on Scientific Computing, 2018, 40(3), C327–C357](https://doi.org/10.1137/17M1129751)  
([arXiv](https://arxiv.org/abs/1804.02338))


Nathan Sime and Cian R. Wilson,  
*Automatic weak imposition of free slip boundary conditions via Nitsche's method: application to
nonlinear problems in geodynamics*  
([arXiv](https://arxiv.org/abs/2001.10639))


## Dependencies

* [UFL](https://github.com/FEniCS/ufl)
* python 3

##### Optional dependencies

* For `dolfin_dg.dolfinx` and the `dolfinx` demos: the core components of the 
  [FEniCSx project](https://fenicsproject.org/).
* For `dolfin_dg.dolfin` and the `dolfin` demos: the core components of the 
  [*legacy* FEniCS project](https://fenicsproject.org/download/archive/).
  * For the hybrid discontinuous Galerkin (HDG) solvers employing static
   condensation with `dolfin`,
   this [LEoPart fork](https://bitbucket.org/nate-sime/leopart/) is required.
* For `firedrake` support the core components of the [Firedrake
  project](https://www.firedrakeproject.org/)


## Installation

Navigate to the `dolfin_dg` directory and install with

```bash
pip install .
```

## Legacy `dolfin` docker image with LEoPart fork

![](https://quay.io/repository/natesime/dolfin_dg/status)

A  docker image facilitating compilation and execution of `dolfin_dg` examples
with the legacy version of `dolfin` and `leopart` is available:

```bash
docker run -it quay.io/natesime/dolfin_dg:master
```


## Contributors

* Nate Sime <nsime@carnegiescience.edu>
* Paul Houston <Paul.Houston@nottingham.ac.uk>
* Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>
* Robert Klöfkorn <robert.klofkorn@math.lu.se>

## License

GNU LGPL, version 3.