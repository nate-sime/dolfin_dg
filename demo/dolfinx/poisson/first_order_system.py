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
            G1 = homogenize(F_0, u, ufl.div(F_1(u)))
            G2 = homogenize(F_1, u, ufl.grad(u))

            # Domain
            F = - ufl.inner(F_1(u), ufl.grad(G_T_mult(G1, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            divibp = DivIBP(F_1, u, v, G1)

            F += divibp.interior_residual1(alpha("+") * ufl.avg(G2), u)
            F += divibp.exterior_residual1(alpha * ufl.replace(G2, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G1, v)), G2)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)
        elif problem_id == 2:
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
            G1 = homogenize(F_0, u, ufl.div(F_1(u)))
            G2 = homogenize(F_1, u, ufl.grad(u))

            # Domain
            F = - ufl.inner(F_1(u), ufl.grad(G_T_mult(G1, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            v1 = v
            divibp = DivIBP(F_1, u, v1, G1)

            v2 = divibp.green_transpose(G_T_mult(G1, v1))
            gradibp = GradIBP(F_2, u, v2, G2)

            F_vec = [divibp, gradibp]
            # fos = dolfin_dg.primal.FirstOrderSystem(F_vec)
            # v3 = gradibp.greens_transpose(G_T_mult(gradibp.G, v2))

            f1 = divibp.interior_residual1(alpha("+") * ufl.avg(G2), u) + \
                 divibp.exterior_residual1(alpha * ufl.replace(G2, {u: u_soln}), u, u_soln, u_soln)
            f2 = gradibp.exterior_residual2(u_soln) + gradibp.interior_residual2()

            n_ibps = [1, 2]
            divgradcurls = [divibp, gradibp]

            m_sign = 1
            F_vec = [f1, f2]

            F_sub = 0
            for j in range(len(F_vec))[::-1]:
                if j < len(F_vec) - 1:
                    if n_ibps[j] == 1 and isinstance(divgradcurls[j], (DivIBP, GradIBP)):
                        F_sub = -1 * F_sub
                F_sub += F_vec[j]
            F += F_sub
        elif problem_id == 3:
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

            F_sub = 0
            for j in range(len(F_vec) - 1)[::-1]:
                if j < len(F_vec) - 2:
                    if n_ibps[j] == 1 and L_vec[j] not in (ufl.div, ufl.grad):
                        F_sub = -1 * F_sub

                IBP = {ufl.div: DivIBP, ufl.grad: GradIBP, ufl.curl: CurlIBP}
                ibp = IBP[L_vec[j]](F_vec[j+1], u, v_vec[j], G_vec[j])
                # print(j, v_vec[j], G_vec[j])
                # quit()

                # print(j, n_ibps[j])
                if n_ibps[j] == 1:
                    # print(j, u, L_vec[j], v_vec[j], G_vec[j])
                    # print(j, ibp, ibp.v, ibp.G, ibp.F)
                    interior = ibp.interior_residual1(alpha("+") * ufl.avg(G_vec[j+1]), u)
                    exterior = ibp.exterior_residual1(alpha * ufl.replace(G_vec[j+1], {u: u_soln}), u, u_soln, u_soln)
                elif n_ibps[j] == 2:
                    # print(j, u, L_vec[j], v_vec[j], G_vec[j])
                    # print(j, ibp, ibp.v, ibp.G, ibp.F)
                    interior = ibp.interior_residual2()
                    exterior = ibp.exterior_residual2(u_soln)
                F_sub += interior + exterior

            F += F_sub
        elif problem_id == 4:
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
            F += fos.interior(alpha, n_ibps, dolfin_dg.primal.facet_sipg, u_soln)

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