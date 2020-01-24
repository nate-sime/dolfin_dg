from dolfin import UnitSquareMesh, MixedElement, VectorElement, \
    FiniteElement, FunctionSpace, FacetNormal, SpatialCoordinate, \
    Function, Constant, split, interpolate, assign, TestFunction, Expression, \
    parameters, solve, errornorm, MPI, \
    dot, inner, div, sym, grad, sqrt, ds, dx, Identity, assemble
import numpy as np

from dolfin_dg import StokesNitscheBoundary, tangential_proj

parameters['std_out_all_processes'] = False
parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 5

ele_ns = [4, 8, 16, 32]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
errorpl2 = np.zeros(len(ele_ns))
errorph1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))

case = 1
if case == 1:
    u_soln_code = ("2*x[1]*(1.0 - x[0]*x[0])",
                   "-2*x[0]*(1.0 - x[1]*x[1])")
elif case == 2:
    u_soln_code = ("-x[1]*sqrt(x[0]*x[0] + x[1]*x[1])",
                   "x[0]*sqrt(x[0]*x[0] + x[1]*x[1])")

for j, ele_n in enumerate(ele_ns):
    mesh = UnitSquareMesh(ele_n, ele_n)
    We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
                       FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = FunctionSpace(mesh, We)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    u_soln = Expression(u_soln_code, degree=4, domain=mesh)
    p_soln = Constant(0.0)

    U = Function(W)
    u, p = split(U)

    uu = interpolate(u_soln, W.sub(0).collapse())
    assign(U.sub(0), uu)

    V = TestFunction(W)
    v, q = split(V)

    def eta(u):
        return 1 + sqrt(inner(grad(u), grad(u)))**-1

    def F_v(u, grad_u, p_local=None):
        if p_local is None:
            p_local = p
        return 2 * eta(u) * sym(grad_u) - p_local * Identity(2)

    f = -div(F_v(u_soln, grad(u_soln), p_soln))
    g_tau = tangential_proj(F_v(u_soln, grad(u_soln), p_soln) * n, n)

    F = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx \
        + div(u) * q * dx

    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q, delta=-1)
    F += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds)

    bcs = []
    solve(F == 0, U)

    uh = U.sub(0, deepcopy=True)
    ph = U.sub(1, deepcopy=True)
    ph.vector()[:] -= assemble(p*dx)

    errorl2[j] = errornorm(u_soln, uh, norm_type='l2', degree_rise=3)
    errorh1[j] = errornorm(u_soln, uh, norm_type='h1', degree_rise=3)
    errorpl2[j] = errornorm(p_soln, ph, norm_type='l2', degree_rise=3)
    errorph1[j] = errornorm(p_soln, ph, norm_type='h1', degree_rise=3)
    hsizes[j] = mesh.hmax()


if MPI.rank(mesh.mpi_comm()) == 0:
    print("L2 u convergence rates: "
          + str(np.log(errorl2[0:-1] / errorl2[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
    print("H1 u convergence rates: "
          + str(np.log(errorh1[0:-1] / errorh1[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
    print("L2 p convergence rates: "
          + str(np.log(errorpl2[0:-1] / errorpl2[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
    print("H1 p convergence rates: "
          + str(np.log(errorph1[0:-1] / errorph1[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
