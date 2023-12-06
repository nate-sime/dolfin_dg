import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl

import dolfin_dg


class ConvergenceTest:

    def __init__(self, meshes, element, TOL=0.1):
        self.meshes = meshes
        self.TOL = TOL
        self.element = element

    def u_soln(self, V):
        pass

    def generate_form(self, mesh, V, u, v):
        pass

    def run_test(self):
        n_runs = len(self.meshes)
        error0 = np.zeros(n_runs)
        hsizes = np.zeros(n_runs)

        run_count = 0
        for mesh in self.meshes:
            V = dolfinx.fem.FunctionSpace(mesh, self.element)
            u, v = dolfinx.fem.Function(V), ufl.TestFunction(V)
            gD = self.u_soln(V)
            F = self.generate_form(mesh, V, u, v)

            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, u, du)

            F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
            problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(
                F, J, u, [])
            snes = PETSc.SNES().create(MPI.COMM_WORLD)
            snes.setFromOptions()
            snes.getKSP().getPC().setType("lu")
            snes.getKSP().getPC().setFactorSolverType("mumps")
            snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
            snes.setJacobian(problem.J_mono,
                             J=dolfinx.fem.petsc.create_matrix(J))
            snes.setTolerances(rtol=1e-14, atol=1e-14)

            snes.solve(None, u.vector)

            error0[run_count] = self.compute_error_norm0(gD, u)

            h_measure = dolfinx.cpp.mesh.h(
                mesh._cpp_object, mesh.topology.dim,
                np.arange(mesh.topology.index_map(mesh.topology.dim).size_local,
                          dtype=np.int32))
            hmax = mesh.comm.allreduce(h_measure.max(), op=MPI.MAX)
            hsizes[run_count] = hmax
            run_count += 1

        rate0 = np.log(error0[0:-1]/error0[1:])/np.log(hsizes[0:-1]/hsizes[1:])

        self.check_norm0_rates(rate0)

    def compute_error_norm0(self, u_soln, u):
        mesh = u.function_space.mesh
        l2error_u = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))**0.5,
            op=MPI.SUM)
        return l2error_u

    def check_norm0_rates(self, rate0):
        expected_rate = float(self.element.degree() + 1)
        assert rate0[0] > expected_rate - self.TOL