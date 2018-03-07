import numpy as np
import ufl

from dolfin import *

from dolfin_dg import *
from dolfin_dg import force_zero_function_derivative
from dolfin_dg.tensors import compressible_ns_entopy_G

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']["quadrature_degree"] = 8
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

def U_to_V(U):
    rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
    i = E - 0.5*(u1**2 + u2**2)
    U1, U2, U3, U4 = U
    s = ln((gamma-1)*rho*i/(U1**gamma))
    V1 = 1/(rho*i)*(-U4 + rho*i*(gamma + 1 - s))
    V2 = 1/(rho*i)*U2
    V3 = 1/(rho*i)*U3
    V4 = 1/(rho*i)*(-U1)
    return as_vector([V1, V2, V3, V4])

def V_to_U(V):
    V1, V2, V3, V4 = V
    U = as_vector([-V4, V2, V3, 1 - 0.5*(V2**2 + V3**2)/V4])
    s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
    rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*exp(-s/(gamma-1))
    U = U*rhoi
    return U

poly_o = 1
run_count = 0
mesh_sizes = [8, 16, 32]

errorl2 = np.array([0]*len(mesh_sizes), dtype=np.double)
errorh1 = np.array([0]*len(mesh_sizes), dtype=np.double)
hsizes = np.array([0]*len(mesh_sizes), dtype=np.double)

for n_eles in mesh_sizes:
    # set up the mesh
    mesh = RectangleMesh(Point(0, 0), Point(pi, pi), n_eles, n_eles, 'right')

    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
    info("DoFs: " + str(V.dim()))
    du = TrialFunction(V)
    v_vec = TestFunction(V)
    n = FacetNormal(mesh)
    h = CellVolume(mesh)/FacetArea(mesh)

    # dirichlet conditions and error suite
    gamma = 1.4
    gD = Expression(('((-std::log((0.4*sin(2*x[0] + 2*x[1]) + 1.6)*pow(sin(2*x[0] + 2*x[1]) + 4, -1.4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4)) + 2.4)*(sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4) - pow(sin(2*x[0] + 2*x[1]) + 4, 2))/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '(-sin(2*x[0] + 2*x[1]) - 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))'),
                    element=V.ufl_element())
    u_vec = interpolate(gD, V)

    # Convection term
    def F_c(V):
        V1, V2, V3, V4 = V
        gamma_h = gamma - 1
        e1 = V2*V4
        e2 = V3*V4
        c1 = gamma_h*V4 - V2**2
        c2 = gamma_h*V4 - V3**2
        d1 = -V2*V3
        k1 = (V2**2 + V3**2)/(2*V4)
        k2 = k1 - gamma

        s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
        rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*exp(-s/(gamma-1))

        res = as_matrix([[e1, e2],
                      [c1, d1],
                      [d1, c2],
                      [k2*V2, k2*V3]])
        res = res*rhoi/V4
        return res

    # Domain part of the convective weak form
    convective_domain = -inner(F_c(u_vec), grad(v_vec))*dx

    # Calculate the maximum of the eigenvalues for the dissipation parameter
    def construct_evs(V, n):
        U = V_to_U(V)
        rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
        p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
        u = as_vector([u1,u2])
        c = sqrt(gamma*p/rho)
        lambdas = [dot(u, n) - c, dot(u, n), dot(u, n), dot(u, n) + c]
        lambdas = list(map(abs,lambdas))
        return lambdas


    def construct_alpha(U_p, U_m, n_p):
        forward_evs = construct_evs(U_p, n_p)
        reverse_evs = construct_evs(U_m, n_p)

        return ufl.Max(
            ufl.Max(ufl.Max(forward_evs[0], forward_evs[1]), forward_evs[2]),
            ufl.Max(ufl.Max(reverse_evs[0], reverse_evs[1]), reverse_evs[2])
        )

    def H(U_p, U_m, n_p):
        U_p_a, U_m_a = force_zero_function_derivative(U_p), force_zero_function_derivative(U_m)
        alpha = construct_alpha(U_p_a, U_m_a, n_p)
        value = Constant(0.5)*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + Constant(1.0)*alpha*(U_p - U_m))
        return value

    # Face terms of the convective weak formulation
    convective_interior = dot(H(u_vec('+'), u_vec('-'), n('+')), (v_vec('+') - v_vec('-')))*dS
    convective_exterior = dot(H(u_vec, gD, n), v_vec)*ds

    convective_residual = convective_domain + convective_interior + convective_exterior

    # Viscous part
    sig = Constant(20.0)*max(poly_o**2, 1)/h
    mu = 1
    lam = -2./3.*mu
    Pr = 0.72

    def F_v(u_vec):
        K = compressible_ns_entopy_G(mu, lam, Pr, gamma, u_vec)
        return hyper_tensor_product(K, grad(u_vec))

    # Viscous domain component
    visc_domain = inner(F_v(u_vec), grad(v_vec))*dx

    # set up utility object for viscous terms
    vt = DGFemSIPG(F_v, u_vec, v_vec, sig, G=compressible_ns_entopy_G(mu, lam, Pr, gamma, u_vec), n=n)

    # Calculate the interior face terms of the viscous component
    visc_interior = vt.interior_residual(dS)

    # Calculate the exterior face terms of the viscous component, u_gamma(u^+) = gD in this case
    # over the whole boundary dS.
    visc_exterior = vt.exterior_residual(gD, ds)

    # Construct the viscous residual
    viscous_residual = visc_domain + visc_interior + visc_exterior

    f = Expression(('0.8*cos(2.0*x[0] + 2.0*x[1])',
                    '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                    '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                    '(2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1]) + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5) + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4) + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3) + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2) + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1]) + 3817.472*sin(4.0*x[0] + 4.0*x[1]) + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 4)'),
                   element=V.ufl_element())

    # Construct convective residual
    residual = convective_residual + viscous_residual - dot(f, v_vec)*dx

    J = derivative(residual, u_vec, du)
    solve(residual == 0, u_vec, [], J=J, solver_parameters={'newton_solver': {'absolute_tolerance': 5e-10}})


    gD_u = Expression(('sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                    element=V.ufl_element())

    u_soln = project(V_to_U(u_vec), V)
    errorl2[run_count] = errornorm(gD_u, u_soln, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD_u, u_soln, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()
    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print(','.join(map(str, errorl2)))
    print(','.join(map(str, errorh1)))
    print(','.join(map(str, hsizes)))
    print(np.log(errorl2[:-1]/errorl2[1:])/np.log(hsizes[:-1]/hsizes[1:]))
    print(np.log(errorh1[:-1]/errorh1[1:])/np.log(hsizes[:-1]/hsizes[1:]))
