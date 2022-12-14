import matplotlib.pyplot as plt
from dolfin import (
    UnitSquareMesh, FunctionSpace, Function, TestFunction, Expression,
    MeshFunction, Measure, dx, Constant, grad, solve, plot, parameters, near,
    AutoSubDomain)

from dolfin_dg import PoissonOperator, DGDirichletBC

# This demo is a reproduction of the nonlinear Poisson; however, using DG FEM.
# http://fenics.readthedocs.io/projects/dolfin/en/stable/demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.html

parameters["ghost_mode"] = "shared_facet"

mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, 'DG', 1)
u, v = Function(V), TestFunction(V)

f = Expression("x[0]*sin(x[1])", degree=1)

facets = MeshFunction("size_t", mesh, 1, 0)
AutoSubDomain(
    lambda x, on_boundary: near(x[0], 1.0) and on_boundary).mark(facets, 1)

ds = Measure("ds", subdomain_data=facets)

g = Constant(1.0)
bc = DGDirichletBC(ds(1), g)

pe = PoissonOperator(mesh, V, [bc], kappa=(1 + u**2))
F = pe.generate_fem_formulation(u, v) - f*v*dx

solve(F == 0, u, solver_parameters={"newton_solver": {"linear_solver": "lu"}})

plot(u)
plot(grad(u))
plt.show()
