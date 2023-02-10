import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx

import dolfin_dg.dolfinx
from dolfin_dg.math import hyper_tensor_T_product as G_T_mult
from dolfin_dg.primal.facet_sipg import DivIBP, GradIBP, CurlIBP
from dolfin_dg.math import homogenize


def pprint(*msg, verbose=False):
    if verbose:
        print(msg)


ele_ns = [8, 16, 32]
# ele_ns = [2, 4]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 3

for problem_id in [4]:
    run_count = 0
    print(f"Running problem ID: {problem_id}")
    for ele_n in ele_ns:
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, ele_n, ele_n,
            cell_type=dolfinx.mesh.CellType.triangle,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            diagonal=dolfinx.mesh.DiagonalType.left)
        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)

        if problem_id == 1:
            # -- Scalar Poisson
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: x[0] + 1)

            u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]) + 1

            # Convective Operator
            def F_2(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_1(u, flux=None):
                if flux is None:
                    flux = ufl.grad(F_2(u))
                return flux

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_1(u))
                return -flux

            f = F_0(u_soln)

            F_vec = [F_0, F_1, F_2]
            L_vec = [ufl.div, ufl.grad]
            G_vec = [homogenize(F_vec[i], u, L_vec[i](F_vec[i+1](u)))
                     for i in range(len(F_vec) - 1)]
            v_vec = [v, *(
                dolfin_dg.primal.green_transpose(L_vec[i])(G_T_mult(G_vec[i], v))
                for i in range(len(F_vec) - 1))]
            n_ibps = [1, 2]

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # Domain
            F = - ufl.inner(F_1(u), v_vec[1]) * ufl.dx - ufl.inner(f, v) * ufl.dx

            fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
            F += fos.interior([alpha], n_ibps, dolfin_dg.primal.facet_sipg, u_soln)
        elif problem_id == 2:
            # -- Vector Poisson
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1)))

            u_soln = ufl.as_vector(
                [ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]) + 1]*2)

            # Convective Operator
            def F_2(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_1(u, flux=None):
                if flux is None:
                    flux = ufl.grad(F_2(u))
                return flux

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_1(u))
                return -flux

            f = F_0(u_soln)

            F_vec = [F_0, F_1, F_2]
            L_vec = [ufl.div, ufl.grad]
            G_vec = [homogenize(F_vec[i], u, L_vec[i](F_vec[i+1](u)))
                     for i in range(len(F_vec) - 1)]
            v_vec = [v]
            for i in range(len(F_vec)-2):
                vj = dolfin_dg.primal.green_transpose(L_vec[i])(G_T_mult(G_vec[i], v_vec[i]))
                v_vec.append(vj)
            n_ibps = [1, 2]

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # Domain
            F = - ufl.inner(F_1(u), v_vec[1]) * ufl.dx - ufl.inner(f, v) * ufl.dx

            fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
            F += fos.interior([alpha], n_ibps, dolfin_dg.primal.facet_sipg, u_soln)
        elif problem_id == 3:
            # -- Maxwell
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1)))

            k = dolfinx.fem.Constant(mesh, 1.0)
            u_soln = ufl.as_vector(
                [ufl.sin(k*x[1]), ufl.sin(k*x[0])])

            # Convective Operator
            def F_2(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_1(u, flux=None):
                if flux is None:
                    flux = ufl.curl(F_2(u))
                return flux

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.curl(F_1(u))
                return flux

            f = F_0(u_soln)

            F_vec = [F_0, F_1, F_2]
            L_vec = [ufl.curl, ufl.curl]
            G_vec = [homogenize(F_vec[i], u, L_vec[i](F_vec[i+1](u)))
                     for i in range(len(F_vec) - 1)]
            v_vec = [v]
            for i in range(len(F_vec)-2):
                vj = dolfin_dg.primal.green_transpose(L_vec[i])(G_T_mult(G_vec[i], v_vec[i]))
                v_vec.append(vj)
            n_ibps = [1, 2]

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # Domain
            F = ufl.inner(F_1(u), v_vec[1]) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx

            fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
            F += fos.interior([alpha], n_ibps, dolfin_dg.primal.facet_sipg, u_soln)
        elif problem_id == 4:
            # -- Biharmonic
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            # u.interpolate(lambda x: x[0] + 1)

            # u_soln = (x[0]*x[1]*(1-x[0])*(1-x[1]))**2
            u_soln = ufl.sin(ufl.pi*x[0])**2 * ufl.sin(ufl.pi*x[1])**2
            # u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
            # u_soln = ufl.exp(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])**2 + 1.0

            def F_4(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_3(u, flux=None):
                if flux is None:
                    flux = ufl.grad(F_4(u))
                return flux

            def F_2(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_3(u))
                return flux

            def F_1(u, flux=None):
                if flux is None:
                    flux = ufl.grad(F_2(u))
                return flux

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_1(u))
                return flux

            f = F_0(u_soln)

            F_vec = [F_0, F_1, F_2, F_3, F_4]
            L_vec = [ufl.div, ufl.grad, ufl.div, ufl.grad]
            G_vec = [homogenize(F_vec[i], u, L_vec[i](F_vec[i+1](u)))
                     for i in range(len(F_vec) - 1)]
            v_vec = [v]
            for i in range(len(F_vec)-2):
                vj = dolfin_dg.primal.green_transpose(L_vec[i])(G_T_mult(G_vec[i], v_vec[i]))
                v_vec.append(vj)
            n_ibps = [1, 1, 2, 2]

            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
            beta = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

            # Domain
            F = ufl.inner(F_2(u), v_vec[2]) * ufl.dx \
                - ufl.inner(f, v) * ufl.dx

            fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
            F += fos.interior([alpha, beta, None, None],
                              n_ibps, dolfin_dg.primal.facet_sipg, u_soln)

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
        # opts["snes_monitor"] = None
        # opts["snes_max_it"] = 1
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