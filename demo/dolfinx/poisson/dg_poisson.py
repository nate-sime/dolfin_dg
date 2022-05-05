import dolfinx.io
import dolfinx.mesh
import dolfinx.fem
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfin_dg.dolfinx

# This demo is a reproduction of the nonlinear Poisson demo, however, using DG
# FEM.
# http://fenics.readthedocs.io/projects/dolfin/en/stable/demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.html

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 32, 32,
    ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

V = dolfinx.fem.FunctionSpace(mesh, ('DG', 1))
u, v = dolfinx.fem.Function(V), ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)
f = x[0]*ufl.sin(x[1])

# Construct boundary measure for BCs
free_end_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, 1, lambda x: np.isclose(x[0], 1.0))
facets = dolfinx.mesh.meshtags(mesh, 1, np.sort(free_end_facets), 1)
ds = ufl.Measure("ds", subdomain_data=facets)

# Boundary condition data
g = dolfinx.fem.Constant(mesh, 1.0)
bc = dolfin_dg.DGDirichletBC(ds(1), g)

# Automated poisson operator DG formulation
pe = dolfin_dg.PoissonOperator(mesh, V, [bc], kappa=(1 + u**2))
F = pe.generate_fem_formulation(u, v) - f*v*ufl.dx

du = ufl.TrialFunction(V)
J = ufl.derivative(F, u, du)

# Setup SNES solver
snes = PETSc.SNES().create(MPI.COMM_WORLD)
opts = PETSc.Options()
opts["snes_monitor"] = None
snes.setFromOptions()
snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")

# Setup nonlinear problem
F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])
snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))

# Solve and plot
snes.solve(None, u.vector)
print(f"SNES converged: {snes.getConvergedReason()}")
print(f"KSP converged: {snes.getKSP().getConvergedReason()}")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "poisson.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)
