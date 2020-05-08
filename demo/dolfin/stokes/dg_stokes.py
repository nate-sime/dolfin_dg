from dolfin import *
from dolfin_dg import *
import numpy as np

parameters['std_out_all_processes'] = False
parameters["ghost_mode"] = "shared_facet"

p_order = 2
l2errors_u = []
l2errors_p = []
hs = []

for n in range(2, 6):
    mesh = UnitSquareMesh(2**n, 2**n, "left")

    Ve = VectorElement("DG", mesh.ufl_cell(), p_order)
    Qe = FiniteElement("DG", mesh.ufl_cell(), p_order-1)
    W = FunctionSpace(mesh, MixedElement([Ve, Qe]))
    info("DoFs: %d" % W.dim())

    U = Function(W)
    u, p = split(U)
    dU = TrialFunction(W)
    V = TestFunction(W)
    v, q = split(V)

    x, y = SpatialCoordinate(mesh)

    u_soln = Expression(("-(x[1]*cos(x[1]) + sin(x[1]))*exp(x[0])",
                           "x[1] * sin(x[1]) * exp(x[0])"),
                        degree=W.sub(0).ufl_element().degree() + 1,
                        domain=mesh)
    p_soln = Expression("2.0 * exp(x[0]) * sin(x[1]) + 1.5797803888225995912 / 3.0",
                        degree=W.sub(1).ufl_element().degree() + 1)

    ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    CompiledSubDomain("near(x[0], 0.0) or near(x[1], 0.0)").mark(ff, 1)
    CompiledSubDomain("near(x[0], 1.0) or near(x[1], 1.0)").mark(ff, 2)
    ds = Measure("ds", subdomain_data=ff)
    dsD, dsN = ds(1), ds(2)

    facet_n = FacetNormal(mesh)
    gN = (grad(u_soln) - p_soln*Identity(2))*facet_n
    bcs = [DGDirichletBC(dsD, u_soln), DGNeumannBC(dsN, gN)]

    def F_v(u, grad_u):
        return grad_u - p*Identity(2)

    stokes = StokesOperator(mesh, W, bcs, F_v)
    F = stokes.generate_fem_formulation(u, v, p, q)
    J = derivative(F, U, dU)

    # Define boundary condition
    solve(F == 0, U, bcs=[], J=J)

    l2error_u = assemble((u - u_soln) ** 2 * dx)**0.5
    l2error_p = assemble((p - p_soln) ** 2 * dx)**0.5

    hs.append(mesh.hmin())
    l2errors_u.append(l2error_u)
    l2errors_p.append(l2error_p)
    info("n: %d, l2 u error: %.6e" % (n, l2error_u))
    info("n: %d, l2 p error: %.6e" % (n, l2error_p))


l2errors_u = np.array(l2errors_u)
l2errors_p = np.array(l2errors_p)
hs = np.array(hs)
rates_u = np.log(l2errors_u[:-1] / l2errors_u[1:]) / np.log(hs[:-1] / hs[1:])
rates_p = np.log(l2errors_p[:-1] / l2errors_p[1:]) / np.log(hs[:-1] / hs[1:])
info("rates u: %s" % str(rates_u))
info("rates p: %s" % str(rates_p))