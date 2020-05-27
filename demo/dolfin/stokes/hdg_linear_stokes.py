import numpy as np
from dolfin import *
import dolfin_dg
import dolfin_dg.hdg_form
import leopart


k = 2
n_eles = [2, 4, 8, 16, 32, 64]
l2errors_u_l2 = np.zeros_like(n_eles, dtype=np.double)
l2errors_u_h1 = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = UnitSquareMesh(n_ele, n_ele)

    u_soln = Expression(("-(x[1]*cos(x[1]) + sin(x[1]))*exp(x[0])",
                         "x[1] * sin(x[1]) * exp(x[0])"),
                        degree=k + 2,
                        domain=mesh)
    p_soln = Expression("2.0 * exp(x[0]) * sin(x[1]) + 1.5797803888225995912 / 3.0",
                        degree=k + 2, domain=mesh)

    ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    CompiledSubDomain("near(x[0], 0.0) or near(x[1], 0.0)").mark(ff, 1)
    CompiledSubDomain("near(x[0], 1.0) or near(x[1], 1.0)").mark(ff, 2)
    ds = Measure("ds", subdomain_data=ff)
    dsD, dsN = ds(1), ds(2)

    Ve = VectorElement("DG", mesh.ufl_cell(), k)
    Vbare = VectorElement("CG", mesh.ufl_cell(), k)["facet"]

    Qe = FiniteElement("DG", mesh.ufl_cell(), k-1)
    Qbare = FiniteElement("DGT", mesh.ufl_cell(), k)

    W = FunctionSpace(mesh, MixedElement([Ve, Qe]))
    Wbar = FunctionSpace(mesh, MixedElement([Vbare, Qbare]))

    U_, Ubar_ = Function(W), Function(Wbar)
    V_, Vbar_ = TestFunction(W), TestFunction(Wbar)

    u, p = split(U_)
    v, q = split(V_)

    ubar, pbar = split(Ubar_)
    vbar, qbar = split(Vbar_)

    alpha = Constant(10.0 * Ve.degree()**2)
    h = CellDiameter(mesh)

    n = FacetNormal(mesh)
    phi = Function(FunctionSpace(mesh, "DG", 1))
    phi.interpolate(Expression("x[0] > 0.5 ? 1.0 : 2.0", degree=1, domain=mesh))
    eta_top = Constant(1.0)
    eta_bottom = Constant(0.0)
    eta = eta_bottom + phi * (eta_top - eta_bottom)

    def facet_integral(integrand):
        return integrand('-') * dS + integrand('+') * dS + integrand * ds

    def F_v(u, grad_u, p_local=None):
        if p_local is None:
            p_local = pbar
        return 2*eta*sym(grad_u) - p_local*Identity(2)

    gN = F_v(u_soln, grad(u_soln), p_soln)*n
    F = inner(F_v(u, grad(u), p), grad(v)) * dx - dot(gN, vbar)*dsN

    penalty = alpha / h
    G = dolfin_dg.homogeneity_tensor(F_v, u)
    hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(F_v, u, ubar, v, vbar, penalty, G, n)

    F += hdg_term.face_residual(dS, ds)

    # Volume source
    f = - div(F_v(u_soln, grad(u_soln), p_soln))
    F += - inner(f, v) * dx

    # Continuity
    F += inner(q, div(u))*dx
    F += facet_integral(inner(dot(u - ubar, n), qbar))

    bcs = [DirichletBC(Wbar.sub(0), u_soln, ff, 1)]

    # Construct local and global block components
    Fr = dolfin_dg.extract_rows(F, [V_, Vbar_])
    J = dolfin_dg.derivative_block(Fr, [U_, Ubar_])

    Fr[0] = -Fr[0]
    Fr[1] = -Fr[1]

    def formit(F):
        if isinstance(F, (list, tuple)):
            return list(map(formit, F))
        return Form(F)

    Fr = formit(Fr)
    J = formit(J)

    # Use static condensation solver
    ssc = leopart.StokesStaticCondensation(
        mesh,
        J[0][0], J[0][1],
        J[1][0], J[1][1],
        Fr[0], Fr[1])

    ssc.assemble_global_system(True)
    for bc in bcs:
        ssc.apply_boundary(bc)
    ssc.solve_problem(Ubar_.cpp_object(), U_.cpp_object(), "mumps", "default")

    l2error_u = assemble(dot(u - u_soln, u - u_soln)*dx)**0.5
    h1error_u = assemble(inner(grad(u - u_soln), grad(u - u_soln))*dx)**0.5

    l2error_p = assemble((p - p_soln)**2*dx)**0.5
    h1error_p = assemble(inner(grad(p - p_soln), grad(p - p_soln))*dx)**0.5

    hs[run_no] = MPI.min(mesh.mpi_comm(), mesh.hmin())
    l2errors_u_l2[run_no] = l2error_u
    l2errors_u_h1[run_no] = h1error_u

rates_u_l2 = np.log(l2errors_u_l2[:-1] / l2errors_u_l2[1:]) / np.log(hs[:-1] / hs[1:])
rates_u_h1 = np.log(l2errors_u_h1[:-1] / l2errors_u_h1[1:]) / np.log(hs[:-1] / hs[1:])
print("rates u L2: %s" % str(rates_u_l2))
print("rates u H1: %s" % str(rates_u_h1))
