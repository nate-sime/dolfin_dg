import numpy as np
from dolfin import *
import dolfin_dg
import dolfin_dg.hdg_form
import leopart


poly_o = 2
n_eles = [8, 16, 32, 64]
l2errors_u_l2 = np.zeros_like(n_eles, dtype=np.double)
l2errors_u_h1 = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = UnitSquareMesh(n_ele, n_ele)
    Ve_high = FiniteElement("CG", mesh.ufl_cell(), poly_o+2)
    Ve = FiniteElement("DG", mesh.ufl_cell(), poly_o)
    Vbare = FiniteElement("CG", mesh.ufl_cell(), poly_o)["facet"]

    u_soln_f = Expression("sin(pi * x[0]) * sin(pi * x[1])", degree=poly_o+2)

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
    b = Constant((1, 1))

    n = FacetNormal(mesh)

    # Second order terms
    kappa = Constant(2.0)

    def F_v(u, grad_u):
        return kappa * grad_u

    F = inner(F_v(u, grad(u)), grad(v)) * dx

    sigma = alpha / h
    G = dolfin_dg.homogeneity_tensor(F_v, u)
    hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(F_v, u, ubar, v, vbar, sigma, G, n)

    F += hdg_term.face_residual(dS, ds)

    # First order terms
    def F_c(u):
        return b * u

    F += -inner(F_c(u), grad(v)) * dx

    H_flux = dolfin_dg.LocalLaxFriedrichs(flux_jacobian_eigenvalues=lambda u, n: 2 * u * dot(b, n))
    hdg_fo_term = dolfin_dg.hdg_form.HDGClassicalFirstOrder(F_c, u, ubar, v, vbar, H_flux, n)

    F += hdg_fo_term.face_residual(dS, ds)

    # Volume source
    f = div(F_c(u_soln) - F_v(u_soln, grad(u_soln)))
    F += - f * v * dx

    bcs = [DirichletBC(Vbar, gDbar, "on_boundary")]

    import dolfin_dg.dolfinx.nls
    Fr = dolfin_dg.dolfinx.nls.extract_rows(F, [v, vbar])
    J = dolfin_dg.dolfinx.nls.derivative_block(Fr, [u, ubar])

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
    time = Timer("ZZZ Stokes solve")
    for bc in bcs:
        ssc.apply_boundary(bc)
    ssc.solve_problem(ubar.cpp_object(), u.cpp_object(), "mumps", "default")

    l2error_u = errornorm(u_soln_f, u, "l2")
    h1error_u = errornorm(u_soln_f, u, "h1")

    hs[run_no] = MPI.min(mesh.mpi_comm(), mesh.hmin())
    l2errors_u_l2[run_no] = l2error_u
    l2errors_u_h1[run_no] = h1error_u

rates_u_l2 = np.log(l2errors_u_l2[:-1] / l2errors_u_l2[1:]) / np.log(hs[:-1] / hs[1:])
rates_u_h1 = np.log(l2errors_u_h1[:-1] / l2errors_u_h1[1:]) / np.log(hs[:-1] / hs[1:])
print("rates u L2: %s" % str(rates_u_l2))
print("rates u H1: %s" % str(rates_u_h1))
