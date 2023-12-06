import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc

import dolfin_dg.dolfinx
import dolfin_dg.penalty
import dolfin_dg.primal
import dolfin_dg.primal.simple


def pprint(*msg, verbose=False):
    if verbose:
        print(msg)


ele_ns = [8, 16, 32, 64]
# ele_ns = [2, 4]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 2

for problem_id in [1, 2, 3, 4, 5, 6, 7]:
    run_count = 0
    print(f"Running problem ID: {problem_id}, p={p}")
    for ele_n in ele_ns:
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, ele_n, ele_n,
            cell_type=dolfinx.mesh.CellType.triangle,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            diagonal=dolfinx.mesh.DiagonalType.left)
        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)
        h = ufl.CellDiameter(mesh)

        if problem_id == 1:
            # -- Scalar Poisson
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)
            u = dolfinx.fem.Function(V, name="u")
            fos = dolfin_dg.primal.simple.diffusion(u, v, A=1)

            u.interpolate(lambda x: x[0] + 1)
            u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]) + 1

            F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
            F += fos.domain()
            alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(
                fos, u, u_soln)
            F += fos.interior([alpha])
            F += fos.exterior([alpha_ext], u_soln)
        elif problem_id == 2:
            # -- Vector Poisson
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)
            u = dolfinx.fem.Function(V, name="u")
            fos = dolfin_dg.primal.simple.diffusion(u, v, 1)

            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1)))
            u_soln = ufl.as_vector(
                [ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]) + 1]*2)

            F = -ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
            F += fos.domain()
            alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(
                fos, u, u_soln)
            F += fos.interior([alpha])
            F += fos.exterior([alpha_ext], u_soln)
        elif problem_id == 3:
            # -- Maxwell
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)
            u = dolfinx.fem.Function(V, name="u")
            fos = dolfin_dg.primal.simple.maxwell(u, v)

            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1)))
            k = dolfinx.fem.Constant(mesh, 1.0)
            u_soln = ufl.as_vector(
                [ufl.sin(k*x[1]), ufl.sin(k*x[0])])

            F = - k**2 * ufl.inner(u, v) * ufl.dx
            F += fos.domain()
            alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(
                fos, u, u_soln)
            F += fos.interior([alpha])
            F += fos.exterior([alpha_ext], u_soln)
        elif problem_id == 4:
            # -- Biharmonic
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)
            u = dolfinx.fem.Function(V, name="u")
            fos = dolfin_dg.primal.simple.biharmonic(u, v)

            u_soln = ufl.sin(ufl.pi*x[0])**2 * ufl.sin(ufl.pi*x[1])**2

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
            beta = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

            F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
            F += fos.domain()
            F += fos.interior([alpha("+") * ufl.avg(fos.G[1]),
                               beta("+") * ufl.avg(fos.G[2])])
            F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln}),
                               beta * ufl.replace(fos.G[2], {u: u_soln})],
                              u_soln)
        elif problem_id == 5:
            # -- Biharmonic
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)
            u = dolfinx.fem.Function(V, name="u")
            mu = dolfinx.fem.Constant(mesh, 1.0)
            fos = dolfin_dg.primal.simple.streamfunction(u, v, mu)

            u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
            beta = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

            F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
            F += fos.domain()
            F += fos.interior([alpha("+") * ufl.avg(fos.G[1]),
                               beta("+") * ufl.avg(fos.G[2])])
            F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln}),
                               beta * ufl.replace(fos.G[2], {u: u_soln})],
                              u_soln)
        elif problem_id == 6:
            # -- Triharmonic
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)
            u = dolfinx.fem.Function(V, name="u")
            fos = dolfin_dg.primal.simple.triharmonic(u, v)
            u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(8 if p <= 2 else 12)) / h**6
            beta = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
            gamma = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

            F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
            F += fos.domain()
            F += fos.interior([alpha("+") * ufl.avg(fos.G[1]),
                               beta("+") * ufl.avg(fos.G[2]),
                               gamma("+") * ufl.avg(fos.G[3])])
            F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln}),
                               beta * ufl.replace(fos.G[2], {u: u_soln}),
                               gamma * ufl.replace(fos.G[3], {u: u_soln})],
                              u_soln)
        elif problem_id == 7:
            # -- Advection diffusion
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u_soln = ufl.sin(ufl.pi*x[0])**2 * ufl.sin(ufl.pi*x[1])**2 + 1

            F, f = 0, 0
            # -- Diffusion
            Ax = dolfinx.fem.Constant(mesh, ((1.0, 0.0),
                                             (0.0, 1.0)))
            fos_diff = dolfin_dg.primal.simple.diffusion(u, v, Ax)
            f += fos_diff.F_vec[0](u_soln)
            F += fos_diff.domain()
            alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(
                fos_diff, u, u_soln)
            F += fos_diff.interior([alpha])
            F += fos_diff.exterior([alpha_ext], u_soln)

            # -- Advection
            b = dolfinx.fem.Constant(mesh, (1.0, 1.0))
            fos_adv = dolfin_dg.primal.simple.advection(u, v, b)
            f += fos_adv.F_vec[0](u_soln)
            F += fos_adv.domain()

            lambdas = ufl.dot(ufl.diff(fos_adv.F_vec[1](u), u), n)
            alpha, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
                lambdas, u, u_soln)
            F += fos_adv.interior([-alpha])
            F += fos_adv.exterior([-alpha_ext], u_soln)

            # -- Reaction
            c = dolfinx.fem.Constant(mesh, 1.0)
            def reaction(u):
                return c * u

            F += ufl.inner(reaction(u), v) * ufl.dx
            f += reaction(u_soln)

            F -= ufl.inner(f, v) * ufl.dx

        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)

        F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
        problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])

        total_dofs = mesh.comm.allreduce(
            V.dofmap.index_map.size_local * V.dofmap.index_map_bs, MPI.SUM)
        if mesh.comm.rank == 0:
            pprint(f"Solving problem: Nele={ele_n}, total DoFs = {total_dofs}")

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        opts = PETSc.Options()
        snes.setFromOptions()
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
        snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
        snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
        snes.setTolerances(rtol=1e-14, atol=1e-14)

        snes.solve(None, u.vector)
        if mesh.comm.rank == 0:
            pprint(f"SNES converged: {snes.getConvergedReason()}")
            pprint(f"KSP converged: {snes.getKSP().getConvergedReason()}")

        l2error_u = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))**0.5,
            op=MPI.SUM)
        errorl2[run_count] = l2error_u

        h1error_u = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form(ufl.grad(u - u_soln) ** 2 * ufl.dx))**0.5,
            op=MPI.SUM)
        errorh1[run_count] = h1error_u

        h_measure = dolfinx.cpp.mesh.h(
            mesh._cpp_object, 2,
            np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32))
        hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
        hsizes[run_count] = hmin

        run_count += 1

    if mesh.comm.rank == 0:
        h_rates = np.log(hsizes[:-1] / hsizes[1:])
        print(f"L2 errors: {errorl2}")
        print(f"L2 error rates: {np.log(errorl2[:-1]/errorl2[1:]) / h_rates}")
        print(f"H1 error rates: {np.log(errorh1[:-1]/errorh1[1:]) / h_rates}")