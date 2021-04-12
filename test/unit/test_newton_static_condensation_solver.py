import numpy as np
from dolfin import (
    UnitSquareMesh, FunctionSpace, Function, TestFunction, dS, dx, errornorm,
    MPI, Expression, Constant, CellDiameter, FacetNormal, inner, grad, dot,
    div, DirichletBC, FiniteElement, MeshFunction, CompiledSubDomain,
    Measure)

import dolfin_dg.dolfin.hdg_newton
import dolfin_dg.hdg_form


def test_hdg_newton_solver_static_condensation():
    poly_o = 1
    n_eles = [8, 16]
    l2errors_u_l2 = np.zeros_like(n_eles, dtype=np.double)
    l2errors_u_h1 = np.zeros_like(n_eles, dtype=np.double)
    hs = np.zeros_like(n_eles, dtype=np.double)


    for run_no, n_ele in enumerate(n_eles):

        mesh = UnitSquareMesh(n_ele, n_ele)
        Ve_high = FiniteElement("CG", mesh.ufl_cell(), poly_o+2)
        Ve = FiniteElement("DG", mesh.ufl_cell(), poly_o)
        Vbare = FiniteElement("CG", mesh.ufl_cell(), poly_o)["facet"]

        ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 2)
        CompiledSubDomain("near(x[0], 0.0) or near(x[1], 0.0)").mark(ff, 1)
        ds = Measure("ds", subdomain_data=ff)
        dsN = ds(2)

        u_soln_f = Expression("1 + sin(pi*(1 + x[0])*pow(1 + x[1], 2) / 8.0)",
                              degree=poly_o+2)

        V = FunctionSpace(mesh, Ve)
        Vbar = FunctionSpace(mesh, Vbare)

        u, ubar = Function(V), Function(Vbar)
        v, vbar = TestFunction(V), TestFunction(Vbar)

        u_soln = Function(FunctionSpace(mesh, Ve_high))
        u_soln.interpolate(u_soln_f)

        gD = Function(V)
        gDbar = Function(Vbar)
        gD.interpolate(u_soln_f)
        gDbar.interpolate(u_soln_f)

        u.interpolate(u_soln)
        ubar.interpolate(u_soln)

        alpha = Constant(10.0 * Ve.degree()**2)
        h = CellDiameter(mesh)
        b = Constant((0.6, 0.8))

        n = FacetNormal(mesh)

        # Second order terms
        def F_v(u, grad_u):
            return (1 + u**2) * grad_u

        F = inner(F_v(u, grad(u)), grad(v)) * dx

        sigma = alpha / h
        G = dolfin_dg.homogeneity_tensor(F_v, u)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, u, ubar, v, vbar, sigma, G, n)

        F += hdg_term.face_residual(dS, ds)

        # Neumann BC
        gN = dot(F_v(u_soln, grad(u_soln)), n)
        F -= gN * vbar * dsN

        # Volume source
        f = -div(F_v(u_soln, grad(u_soln)))
        F += - f * v * dx

        bcs = [DirichletBC(Vbar, gDbar, ff, 1)]

        Fr = dolfin_dg.extract_rows(F, [v, vbar])
        J = dolfin_dg.derivative_block(Fr, [u, ubar])

        solver = dolfin_dg.dolfin.hdg_newton.StaticCondensationNewtonSolver(
            Fr, J, bcs)
        solver.solve(u, ubar)

        l2error_u = errornorm(u_soln_f, u, "l2")
        h1error_u = errornorm(u_soln_f, u, "h1")

        hs[run_no] = MPI.min(mesh.mpi_comm(), mesh.hmin())
        l2errors_u_l2[run_no] = l2error_u
        l2errors_u_h1[run_no] = h1error_u

    hrates = np.log(hs[:-1] / hs[1:])
    rates_u_l2 = np.log(l2errors_u_l2[:-1] / l2errors_u_l2[1:]) / hrates
    rates_u_h1 = np.log(l2errors_u_h1[:-1] / l2errors_u_h1[1:]) / hrates

    assert np.all(rates_u_l2 > poly_o + 1.0 - 0.1)
    assert np.all(rates_u_h1 > poly_o - 0.1)
