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

for problem_id in [1, 2, 3, 4, 5, 6, 7, 8]:
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
            # -- Linear advection
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: x[0] + 1)

            u_soln = ufl.exp(x[0] - x[1])
            # f = dolfinx.fem.Constant(mesh, np.array(0, dtype=np.double))
            b = dolfinx.fem.Constant(mesh, np.array((1, 1), dtype=np.double))

            # Convective Operator
            def F_c(U, flux=None):
                if flux is None:
                    flux = b*U
                return flux

            def F_0(U, flux=None):
                if flux is None:
                    flux = ufl.div(F_c(U))
                return flux

            # f = ufl.div(F_c(u_soln))
            f = F_0(u_soln)

            G0 = homogenize(F_0, u, ufl.div(F_c(u)))

            # Domain
            F = - ufl.inner(F_c(u), ufl.grad(G_T_mult(G0, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            eigen_vals_max_p = abs(ufl.dot(ufl.diff(F_c(u), u), n)("+"))
            eigen_vals_max_m = abs(ufl.dot(ufl.diff(F_c(u), u), n)("-"))
            alpha = dolfin_dg.math.max_value(eigen_vals_max_p, eigen_vals_max_m) / 2.0

            divibp = DivIBP(F_c, u, v, G0)
            F += divibp.interior_residual1(-alpha, u)

            # Exterior
            eigen_vals_max_p = abs(ufl.dot(ufl.diff(F_c(u), u), n))
            u_soln_var = ufl.variable(u_soln)
            eigen_vals_max_m = abs(ufl.dot(ufl.diff(F_c(u_soln_var), u_soln_var), n))
            alpha = dolfin_dg.math.max_value(eigen_vals_max_p, eigen_vals_max_m) / 2.0
            F += divibp.exterior_residual1(-alpha, u, u_soln, u_soln)
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
            G0 = homogenize(F_0, u, ufl.div(F_1(u)))
            G1 = homogenize(F_1, u, ufl.grad(u))

            # Domain
            F = - ufl.inner(F_1(u), ufl.grad(G_T_mult(G0, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            divibp = DivIBP(F_1, u, v, G0)

            F += divibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += divibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G0, v)), G1)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)
        elif problem_id == 3:
            # -- Vector Poisson
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1.0)))

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
            G0 = homogenize(F_0, u, ufl.div(F_1(u)))
            G1 = homogenize(F_1, u, ufl.grad(u))

            # Domain
            F = -ufl.inner(F_1(u), ufl.grad(G_T_mult(G0, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # F0(u, F1(u)) = -div(F1(u))
            divibp = DivIBP(F_1, u, v, G0)
            F += divibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += divibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G0, v)), G1)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)
        elif problem_id == 4:
            # -- Linear elasticity
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1.0)))
            u_soln = ufl.as_vector(
                [ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])]*2)

            E = 1e9
            nu = 0.3
            mu = dolfinx.fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
            lmda = dolfinx.fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

            # Convective Operator
            def F_2(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_1(u, grad_u=None):
                if grad_u is None:
                    grad_u = ufl.grad(F_2(u))
                return 2 * mu * ufl.sym(grad_u) + lmda * ufl.tr(ufl.sym(grad_u)) * ufl.Identity(2)

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_1(u))
                return -flux

            f = F_0(u_soln)
            G0 = homogenize(F_0, u, ufl.div(F_1(u)))
            G1 = homogenize(F_1, u, ufl.grad(u))

            # Domain
            F = -ufl.inner(F_1(u), ufl.grad(G_T_mult(G0, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # F0(u, F1(u)) = -div(F1(u))
            divibp = DivIBP(F_1, u, v, G0)
            F += divibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += divibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G0, v)), G1)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)
        elif problem_id == 5:
            # -- Linear elasticity grad div
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1.0)))
            u_soln = ufl.as_vector(
                [ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])]*2)

            E = 1.0
            nu = 0.3
            mu = dolfinx.fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
            lmda = dolfinx.fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

            # -- Grad Div
            # Convective Operator
            def F_2(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_1(u, div_u=None):
                if div_u is None:
                    div_u = ufl.div(F_2(u))
                return (lmda + mu) * div_u

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.grad(F_1(u))
                return -flux

            f = F_0(u_soln)
            G0 = homogenize(F_0, u, ufl.grad(F_1(u)))
            G1 = homogenize(F_1, u, ufl.div(u))

            # Domain
            F = -ufl.inner(F_1(u), ufl.div(G_T_mult(G0, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # F0(u, F1(u)) = -div(F1(u))
            gradibp = GradIBP(F_1, u, v, G0)
            F += gradibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += gradibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            divibp = DivIBP(F_2, u, ufl.div(G_T_mult(G0, v)), G1)
            F -= divibp.interior_residual2()
            F -= divibp.exterior_residual2(u_soln)

            # -- Div Grad
            # Convective Operator
            def F_2(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_1(u, grad_u=None):
                if grad_u is None:
                    grad_u = ufl.grad(F_2(u))
                return 2 * mu * grad_u

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_1(u))
                return -flux

            f = F_0(u_soln)
            G0 = homogenize(F_0, u, ufl.div(F_1(u)))
            G1 = homogenize(F_1, u, ufl.grad(u))

            # Domain
            F += -ufl.inner(F_1(u), ufl.grad(G_T_mult(G0, v))) * ufl.dx - ufl.inner(f, v) * ufl.dx

            # Interior
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # F0(u, F1(u)) = -div(F1(u))
            divibp = DivIBP(F_1, u, v, G0)
            F += divibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += divibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G0, v)), G1)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)
        elif problem_id == 6:
            # -- Maxwell
            V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1.0)))

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
            G0 = homogenize(F_0, u, ufl.curl(F_1(u)))
            G1 = homogenize(F_1, u, ufl.curl(u))

            # Domain
            F = ufl.inner(F_1(u), ufl.curl(G_T_mult(G0, v))) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx

            # Interior
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

            # F0(u, F1(u)) = curl(F1(u))
            curl1ibp = CurlIBP(F_1, u, v, G0)
            F += curl1ibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += curl1ibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 curl(F2(u))
            curl2ibp = CurlIBP(F_2, u, ufl.curl(G_T_mult(G0, v)), G1)
            F += curl2ibp.interior_residual2()
            F += curl2ibp.exterior_residual2(u_soln)
        elif problem_id == 7:
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

            # Homogenisers
            G0 = homogenize(F_0, u, ufl.div(F_1(u)))
            G1 = homogenize(F_1, u, ufl.grad(F_2(u)))
            G2 = homogenize(F_2, u, ufl.div(F_3(u)))
            G3 = homogenize(F_3, u, ufl.grad(F_4(u)))

            # Domain
            F = ufl.inner(F_2(u), ufl.div(G_T_mult(G1, ufl.grad(G_T_mult(G0, v))))) * ufl.dx \
                - ufl.inner(f, v) * ufl.dx

            # Interior
            # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
            beta = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

            # F0(u, F1(u)) = G0 div(F1(u))
            divibp = DivIBP(F_1, u, v, G0)
            F += divibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += divibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 grad(F2(u))
            gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G0, v)), G1)
            F -= gradibp.interior_residual1(beta("+") * ufl.avg(G2), ufl.grad(u))
            F -= gradibp.exterior_residual1(
                beta * ufl.replace(G2, {u: u_soln}), ufl.grad(u), ufl.grad(u_soln), u_soln)

            # F2(u, F3(u)) = G2 div(F3(u))
            divibp = DivIBP(F_3, u, ufl.div(G_T_mult(G1, ufl.grad(G_T_mult(G0, v)))), G2)
            F += divibp.interior_residual2()
            F += divibp.exterior_residual2(u_soln)

            # F3(u) = G3 grad(F4(u))
            gradibp = GradIBP(F_4, u, ufl.grad(G_T_mult(G2, ufl.div(G_T_mult(G1, ufl.grad(G_T_mult(G0, v)))))), G3)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)
        elif problem_id == 8:
            # -- Biharmonic
            V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
            v = ufl.TestFunction(V)

            u = dolfinx.fem.Function(V, name="u")
            # u.interpolate(lambda x: x[0] + 1)

            # u_soln = x[0]**2 + x[1]
            # u_soln = (x[0]*x[1]*(1-x[0])*(1-x[1]))**2
            # u_soln = ufl.sin(ufl.pi*x[0])**2 * ufl.sin(ufl.pi*x[1])**2
            u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
            # u_soln = ufl.exp(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])**2 + 1.0
            mu = dolfinx.fem.Constant(mesh, 1.0)

            def F_4(u, flux=None):
                if flux is None:
                    flux = u
                return flux

            def F_3(u, flux=None):
                if flux is None:
                    flux = ufl.curl(F_4(u))
                return flux

            def F_2(u, flux=None):
                if flux is None:
                    flux = ufl.grad(F_3(u))
                return mu * (flux + flux.T)

            def F_1(u, flux=None):
                if flux is None:
                    flux = ufl.div(F_2(u))
                return flux

            def F_0(u, flux=None):
                if flux is None:
                    flux = ufl.curl(F_1(u))
                return -flux

            f = F_0(u_soln)

            # Homogenisers
            G0 = homogenize(F_0, u, ufl.curl(F_1(u)))
            G1 = homogenize(F_1, u, ufl.div(F_2(u)))
            G2 = homogenize(F_2, u, ufl.grad(F_3(u)))
            G3 = homogenize(F_3, u, ufl.curl(F_4(u)))

            # Domain
            F = - ufl.inner(F_2(u), ufl.grad(G_T_mult(G1, ufl.curl(G_T_mult(G0, v))))) * ufl.dx \
                - ufl.inner(f, v) * ufl.dx

            # Interior
            # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
            h = ufl.CellDiameter(mesh)
            alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
            beta = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

            # F0(u, F1(u)) = G0 curl(F1(u))
            curl1ibp = CurlIBP(F_1, u, v, G0)
            F += curl1ibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
            F += curl1ibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

            # F1(u) = G1 div(F2(u))
            divibp = DivIBP(F_2, u, ufl.curl(G_T_mult(G0, v)), G1)
            F += divibp.interior_residual1(beta("+") * ufl.avg(G2), ufl.curl(u))
            F += divibp.exterior_residual1(
                beta * ufl.replace(G2, {u: u_soln}), ufl.curl(u), ufl.curl(u_soln), u_soln)

            # F2(u, F3(u)) = G2 grad(F3(u))
            gradibp = GradIBP(F_3, u, ufl.grad(G_T_mult(G1, ufl.curl(G_T_mult(G0, v)))), G2)
            F -= gradibp.interior_residual2()
            F -= gradibp.exterior_residual2(u_soln)

            # F3(u) = G3 curl(F4(u))
            curl2ibp = CurlIBP(F_4, u, ufl.div(G_T_mult(G2, ufl.grad(G_T_mult(G1, ufl.curl(G_T_mult(G0, v)))))), G3)
            F += curl2ibp.interior_residual2()
            F += curl2ibp.exterior_residual2(u_soln)


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