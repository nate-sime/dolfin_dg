import matplotlib.pyplot as plt
import numpy as np
import ufl
import dolfinx
import dolfinx.plotting
from dolfinx.fem.assemble import assemble_matrix, assemble_vector
import dolfin_dg
from dolfin_dg.dolfinx import GenericSNESProblem, MatrixType

from petsc4py import PETSc
from mpi4py import MPI

__author__ = 'njcs4'

# This demo is a reproduction of the nonlinear Poisson;
# however, using DG FEM.
# http://fenics.readthedocs.io/projects/dolfin/en/stable/demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.html

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 32, 32)

V = dolfinx.FunctionSpace(mesh, ('DG', 1))
u, v = dolfinx.Function(V), ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)
f = x[0]*ufl.sin(x[1])

free_end_facets = dolfinx.mesh.locate_entities_geometrical(mesh, 1, lambda x: np.isclose(x[0], 1.0), boundary_only=True)
facets = dolfinx.mesh.MeshTags(mesh, 1, free_end_facets, 1)
ds = ufl.Measure("ds", subdomain_data=facets)

g = dolfinx.Constant(mesh, 1.0)
bc = dolfin_dg.DGDirichletBC(ds(1), g)

pe = dolfin_dg.PoissonOperator(mesh, V, [bc], kappa=(1 + u**2))
F = pe.generate_fem_formulation(u, v) - f*v*ufl.dx

du = ufl.TrialFunction(V)
J = ufl.derivative(F, u, du)

problem = GenericSNESProblem(J, F, None, [], u)
snes = PETSc.SNES().create(MPI.COMM_WORLD)

opts = PETSc.Options()
opts["snes_monitor"] = None
snes.setFromOptions()

snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")

b_vec = dolfinx.fem.create_vector(F)
A_mat = dolfinx.fem.create_matrix(J)

snes.setFunction(problem.F, b_vec)
snes.setJacobian(problem.J, J=A_mat)

snes.solve(None, u.vector)

print(snes.getConvergedReason())
print(snes.getKSP().getConvergedReason())

dolfinx.plotting.plot(u)
plt.show()