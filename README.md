# dolfin_dg

## What does it do?

`dolfin_dg` provides utility functions for the automatic generation of nonlinear
DG FEM formulations using [UFL](https://github.com/FEniCS/ufl).


## Specifically, what does it do?

Consider the nonlinear conservation equation

[//]: <> (\nabla \cdot \mathcal{L}(u; \nabla u) = f)
![](https://latex.codecogs.com/gif.download?-%5Cnabla%20%5Ccdot%20%5Cmathcal%7BL%7D%28u%3B%20%5Cnabla%20u%29%20%3D%20f)

[//]: <> (\mathcal{L}(\cdot; \cdot))
[//]: <> (u \in V)

[nonlinearoperator]: https://latex.codecogs.com/gif.download?%5Cmathcal%7BL%7D%28%5Ccdot%3B%20%5Ccdot%29
[uinv]: https://latex.codecogs.com/gif.download?u%20%5Cin%20V

Here ![][nonlinearoperator]
is a semilinear operator (nonlinear in the first argument and linear in the
second). The semilinear residual weak formulation, subject to appropriate
boundary conditions, reads: find ![][uinv] such that

[//]: <> (\mathcal{N}(u; v) = \int_D \mathcal{L}(u; \nabla u) : \nabla v \; \mathrm{d}x - \int_{\partial D} \mathcal{L}(u; \nabla u) \cdot n \cdot v \; \mathrm{d} s - \int_D f \cdot v \; \mathrm{d} x \equiv 0 \quad \forall v \in V.)

![](https://latex.codecogs.com/gif.download?%5Cmathcal%7BN%7D%28u%3B%20v%29%20%3A%3D%20%5C%5C%20%5Cint_D%20%5Cmathcal%7BL%7D%28u%3B%20%5Cnabla%20u%29%20%3A%20%5Cnabla%20v%20%5C%3B%20%5Cmathrm%7Bd%7Dx%20-%20%5Cint_%7B%5Cpartial%20D%7D%20%5Cmathcal%7BL%7D%28u%3B%20%5Cnabla%20u%29%20%5Ccdot%20n%20%5Ccdot%20v%20%5C%3B%20%5Cmathrm%7Bd%7D%20s%20-%20%5Cint_D%20f%20%5Ccdot%20v%20%5C%3B%20%5Cmathrm%7Bd%7D%20x%20%5Cequiv%200%20%5Cquad%20%5Cforall%20v%20%5Cin%20V.)

[uhinvh]: https://latex.codecogs.com/gif.download?u_h%20%5Cin%20V_h

The DG FEM formulation of the above equation is notoriously verbose to define.
Programming the code to compute the DG FEM approximation ![][uhinvh] is a large
task. `dolfin_dg` provides utility functions for the automatic formulation of
the DG FEM discretisation of the weak formulation. These functions work with
UFL to facilitate simple implementation and efficient computation of the DG
approximation of nonlinear FEM problems.


## More details and citing

Paul Houston and Nathan Sime,  
*Automatic symbolic computation for discontinuous Galerkin finite element methods*,  
[SIAM Journal on Scientific Computing, 2018, 40(3), C327–C357](https://doi.org/10.1137/17M1129751).  
([arXiv](https://arxiv.org/abs/1804.02338))


Nathan Sime and Cian R. Wilson,  
*Automatic weak imposition of free slip boundary conditions via Nitsche's method: application to
nonlinear problems in geodynamics*.  
([arXiv](https://arxiv.org/abs/2001.10639))


## Dependencies

* [UFL](https://github.com/FEniCS/ufl)
* python 3

##### Optional dependencies

* For `dolfin` support and the example demos: the core components of the [FEniCS
  project](https://fenicsproject.org/).
* For `firedrake` support the core components of the [Firedrake
  project](https://www.firedrakeproject.org/)


## Automated testing

[![Pipelines Build Status](https://img.shields.io/bitbucket/pipelines/nate-sime/dolfin_dg)](https://bitbucket.org/nate-sime/dolfin_dg/addon/pipelines/home)

Unit tests are provided in ``test/unit/test_*.py``.


## Installation

#### Docker image


![](https://quay.io/repository/natesime/dolfin_dg/status)


Follow the instructions for installing https://fenicsproject.org/download/. A docker image 
of the `dolfin_dg` master branch is available:


```bash
docker run -it quay.io/natesime/dolfin_dg:master
```

#### Custom installation

Install in the usual way with your preferred prefix:

```bash
python3 setup.py install --prefix=$HOME/local
```


#### Installation inside existing docker container

Follow the instructions to install FEniCS with docker https://fenicsproject.org/download/.

Run docker and clone `dolfin_dg`

```bash
docker run -it quay.io/fenicsproject/dev
git clone https://bitbucket.org/nate-sime/dolfin_dg.git
```

Install `dolfin_dg`

```bash
cd dolfin_dg
sudo python3 setup.py install
```


## Contributors

* Nate Sime <nsime@carnegiescience.edu>
* Paul Houston <Paul.Houston@nottingham.ac.uk>
* Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>


## License

GNU LGPL, version 3.