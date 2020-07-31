import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from dolfin_dg import *

__author__ = 'njcs4'

parameters["ghost_mode"] = "shared_facet"

# a = max distance
# d = max time
a, d = 1.0, 0.2


class FixedBC(SubDomain):
    def inside(self, x, on):
        return (abs(x[0]) < DOLFIN_EPS or abs(x[1]) < DOLFIN_EPS) and on


class FreeBC(SubDomain):
    def inside(self, x, on):
        return (abs(x[0] - a) < DOLFIN_EPS or abs(x[1] - d) < DOLFIN_EPS) and on


# Mesh and function space.
mesh = RectangleMesh(Point(0.0, 0.0), Point(a, d), 64, 64)
V = FunctionSpace(mesh, 'DG', 1)
du = TrialFunction(V)

# Set up Dirichlet BC
gD = Expression('max(sin(2.0*pi*x[0]/a), 0.0)',
                a=a, element=V.ufl_element())
u = project(gD, V)
v = TestFunction(V)

# Establish exterior boundary components and construct measure
exterior_bdries = MeshFunction('size_t', mesh,
                               mesh.topology().dim()-1, 0)
FixedBC().mark(exterior_bdries, 1)
FreeBC().mark(exterior_bdries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=exterior_bdries)

bcs = [DGDirichletBC(ds(1), gD), DGNeumannBC(ds(2), u)]

# List of fluxes to be used in the comparison
fluxes = [
    ("Vijayasundaram",
     Vijayasundaram(lambda u, n: 0.5 * u * n[0] + n[1],
                    lambda u, n: 1, lambda u, n: 1)),
    ("local-Lax Friedrichs",
     LocalLaxFriedrichs(lambda u, n: u * n[0] + n[1])),
    ("HLLE",
     HLLE(lambda u, n: u * n[0] + n[1]))
]

for name, flux in fluxes:
    project(gD, mesh=mesh, function=u)

    # Automatic formulation of the DG spacetime Burgers problem
    bo = SpacetimeBurgersOperator(mesh, V, bcs, flux=flux)
    residual = bo.generate_fem_formulation(u, v)
    J = derivative(residual, u, du)

    solve(residual == 0, u)

    x = np.linspace(0.0, 1.0, 251)
    u_xy = [u(pt, d) for pt in x]
    plt.plot(x, u_xy)

plt.legend([name for name, _ in fluxes])
plt.xlabel("$x$")
plt.ylabel("$u_h(x)$")

plot(u)
plt.show()