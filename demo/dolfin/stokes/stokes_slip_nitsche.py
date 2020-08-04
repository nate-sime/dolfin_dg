from dolfin import UnitSquareMesh, MixedElement, VectorElement, \
    FiniteElement, FunctionSpace, FacetNormal, SpatialCoordinate, \
    Function, Constant, split, interpolate, assign, TestFunction, Expression, \
    parameters, solve, errornorm, MPI, \
    dot, inner, div, sym, grad, sqrt, ds, dx, Identity, assemble, RectangleMesh, Point
import numpy as np

from dolfin_dg import StokesNitscheBoundary, tangential_proj, normal_proj

"""
This demo is the converted (old) dolfin implementation of Example 1
presented in Sime & Wilson (2020) https://arxiv.org/abs/2001.10639
which was originally written for dolfin-x.

To see close to optimal rates of convergence in case 2, we recommend
using higher order quadrature and an element degree approximation of
the true solution > p + 4. These parameters lead to long compilation
times and are switched off by default so that CI tests are not so
expensive.
"""

parameters['std_out_all_processes'] = False
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
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), ele_n, ele_n)
    We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
                       FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = FunctionSpace(mesh, We)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    # Define the velocity solution on an appropriately precise element
    u_soln = Expression(u_soln_code, degree=4, domain=mesh)
    p_soln = Constant(0.0)

    U = Function(W)
    u, p = split(U)

    V = TestFunction(W)
    v, q = split(V)

    # Prevent initial singularity in viscosity
    uu = interpolate(u_soln, W.sub(0).collapse())
    assign(U.sub(0), uu)

    # Nonlinear viscosity model
    def eta(u):
        return 1 + sqrt(inner(grad(u), grad(u)))**-1

    # Problem definition in terms of a viscous flux operator F_v(u, grad(u))
    def F_v(u, grad_u, p_local=None):
        if p_local is None:
            p_local = p
        return 2 * eta(u) * sym(grad_u) - p_local * Identity(2)

    f = -div(F_v(u_soln, grad(u_soln), p_soln))
    g_tau = tangential_proj(F_v(u_soln, grad(u_soln), p_soln) * n, n)

    F = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx \
        + div(u) * q * dx

    # The following two methods for formulating the Nitsche BC are equivalent
    use_bc_utility_method = True

    if use_bc_utility_method == 1:
        # Use the utiltiy method of the StokesNitscheBoundary class
        def F_v_n(u, grad_u, p_local=None):
            return F_v(u, grad_u, p_local)
        stokes_nitsche = StokesNitscheBoundary(F_v_n, u, p, v, q, delta=-1)
        F += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds)
    else:
        # Define the viscous flux in the normal direction
        def F_v_n(u, grad_u, p_local=None):
            return normal_proj(F_v(u, grad_u, p_local), n)
        stokes_nitsche = StokesNitscheBoundary(F_v_n, u, p, v, q, delta=-1)
        F += stokes_nitsche.nitsche_bc_residual(tangential_proj(u, n), ds) \
             - dot(g_tau, v)*ds

    solve(F == 0, U)

    uh = U.sub(0, deepcopy=True)
    ph = U.sub(1, deepcopy=True)

    # Normalise the pressure
    mesh_area = assemble(Constant(1.0)*dx(domain=mesh))
    ph.vector()[:] -= assemble(ph*dx)/mesh_area

    # Compute error
    errorl2[j] = errornorm(u_soln, uh, norm_type='l2', degree_rise=3)
    errorh1[j] = errornorm(u_soln, uh, norm_type='h1', degree_rise=3)
    errorpl2[j] = errornorm(p_soln, ph, norm_type='l2', degree_rise=3)
    errorph1[j] = errornorm(p_soln, ph, norm_type='h1', degree_rise=3)
    hsizes[j] = mesh.hmax()


# Output convergence rates
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
