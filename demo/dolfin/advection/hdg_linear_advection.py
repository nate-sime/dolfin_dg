import leopart
import numpy as np
from dolfin import (
    Point, FunctionSpace, TestFunction, Expression, Constant, inner, grad, dot,
    FacetNormal, MPI, Function, dS, dx, ds, errornorm, RectangleMesh,
    FiniteElement, CellDiameter, div, Form)

import dolfin_dg.hdg_form

poly_o = 2
n_eles = [8, 16, 32, 64]
l2errors_u_l2 = np.zeros_like(n_eles, dtype=np.double)
l2errors_u_h1 = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), n_ele, n_ele, "left/right")
    Ve_high = FiniteElement("CG", mesh.ufl_cell(), poly_o+2)
    Ve = FiniteElement("DG", mesh.ufl_cell(), poly_o)
    Vbare = FiniteElement("CG", mesh.ufl_cell(), poly_o)["facet"]

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

    alpha = Constant(10.0 * Ve.degree()**2)
    h = CellDiameter(mesh)

    n = FacetNormal(mesh)

    a = Constant((0.8, 0.6))
    g = -dot(a, n)*gD

    def F_c(u):
        return a*u

    # Volume source
    f = div(a*gD)
    F = -inner(F_c(u), grad(v))*dx - f*v*dx

    H_flux = dolfin_dg.LocalLaxFriedrichs(
        flux_jacobian_eigenvalues=lambda u, n: dot(a, n))
    hdg_fo_term = dolfin_dg.hdg_form.HDGClassicalFirstOrder(
        F_c, u, ubar, v, vbar, H_flux, n,
        dg_bcs=[dolfin_dg.DGDirichletBC(ds, gD)])

    F += hdg_fo_term.face_residual(dS, ds)

    Fr = dolfin_dg.extract_rows(F, [v, vbar])
    J = dolfin_dg.derivative_block(Fr, [u, ubar])

    Fr[0] = -Fr[0]
    Fr[1] = -Fr[1]

    def formit(F):
        if isinstance(F, (list, tuple)):
            return list(map(formit, F))
        return Form(F)

    Fr = formit(Fr)
    J = formit(J)

    ssc = leopart.StokesStaticCondensation(
        mesh,
        J[0][0], J[0][1],
        J[1][0], J[1][1],
        Fr[0], Fr[1])

    ssc.assemble_global_system(True)
    ssc.solve_problem(ubar.cpp_object(), u.cpp_object(), "mumps", "default")

    l2error_u = errornorm(u_soln_f, u, "l2")
    h1error_u = errornorm(u_soln_f, u, "h1")

    hs[run_no] = MPI.min(mesh.mpi_comm(), mesh.hmin())
    l2errors_u_l2[run_no] = l2error_u
    l2errors_u_h1[run_no] = h1error_u

hrates = np.log(hs[:-1] / hs[1:])
rates_u_l2 = np.log(l2errors_u_l2[:-1] / l2errors_u_l2[1:]) / hrates
rates_u_h1 = np.log(l2errors_u_h1[:-1] / l2errors_u_h1[1:]) / hrates
print(l2errors_u_l2)
print("rates u L2: %s" % str(rates_u_l2))
print("rates u H1: %s" % str(rates_u_h1))
