import numpy as np
import ufl
import dolfinx
from dolfinx.fem.assemble import assemble_matrix, assemble_vector
import dolfin_dg

from petsc4py import PETSc
from mpi4py import MPI

__author__ = 'njcs4'

# This demo is a reproduction of the nonlinear Poisson;
# however, using DG FEM.
# http://fenics.readthedocs.io/projects/dolfin/en/stable/demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.html


class PoissonProblem(dolfinx.NonlinearProblem):
    def __init__(self, a, L):
        super().__init__()
        self.L = dolfinx.Form(L)
        self.a = dolfinx.Form(a)
        self._F = None
        self._J = None

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x):
        if self._F is None:
            self._F = assemble_vector(self.L)
        else:
            with self._F.localForm() as f_local:
                f_local.set(0.0)
            self._F = assemble_vector(self._F, self.L)
        self._F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return self._F

    def J(self, x):
        if self._J is None:
            self._J = assemble_matrix(self.a)
        else:
            self._J.zeroEntries()
            self._J = assemble_matrix(self._J, self.a)
        self._J.assemble()
        return self._J



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

# du = ufl.TrialFunction(V)
# J = ufl.derivative(F, u, du)

dolfinx.Form(F)
# problem = PoissonProblem(J, F)
# solver = dolfinx.NewtonSolver(MPI.COMM_WORLD)
