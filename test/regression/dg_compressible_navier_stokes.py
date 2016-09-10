from dolfin import *
from dolfin_dg import *
import numpy as np

poly_o = 1
parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

run_count = 0
mesh_sizes = [4, 8, 16, 32, 64]

errorl2 = np.array([0]*len(mesh_sizes), dtype=np.double)
errorh1 = np.array([0]*len(mesh_sizes), dtype=np.double)
hsizes = np.array([0]*len(mesh_sizes), dtype=np.double)

for n_eles in mesh_sizes:
    mesh = RectangleMesh(Point(0.0, 0.0), Point(pi, pi), n_eles, n_eles, 'left/right')

    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
    du = TrialFunction(V)
    v_vec = TestFunction(V)

    gamma = Constant(1.4)
    gD = Expression(('sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                    element=V.ufl_element())

    u_vec = project(gD, V)
    n = FacetNormal(mesh)

    # Convective component

    def F_c(U):
        rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
        p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
        H = E + p/rho
        res = as_matrix([[rho*u1, rho*u2],
                      [rho*u1**2 + p, rho*u1*u2],
                      [rho*u1*u2, rho*u2**2 + p],
                      [rho*H*u1, rho*H*u2]])
        return res

    convective_domain = -inner(F_c(u_vec), grad(v_vec))*dx

    def construct_evs(U, n):
        rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
        p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
        u = as_vector([u1, u2])
        c = sqrt(gamma*p/rho)
        lambdas = [dot(u,n) - c, dot(u,n), dot(u,n) + c]
        lambdas = map(abs, lambdas)
        return lambdas

    def construct_alpha(U_p, U_m, n_p):
        forward_evs = construct_evs(U_p, n_p)
        reverse_evs = construct_evs(U_m, n_p)

        return Max(
            Max(Max(forward_evs[0], forward_evs[1]), forward_evs[2]),
            Max(Max(reverse_evs[0], reverse_evs[1]), reverse_evs[2])
        )

    def H(U_p, U_m, n_p):
        U_p_a, U_m_a = force_zero_function_derivative(U_p), force_zero_function_derivative(U_m)
        alpha = construct_alpha(U_p_a, U_m_a, n_p)
        value = Constant(0.5)*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + Constant(1.0)*alpha*(U_p - U_m))
        return value

    convective_interior = dot(H(u_vec('+'), u_vec('-'), n('+')), (v_vec('+') - v_vec('-')))*dS

    convective_exterior = dot(H(u_vec, gD, n), v_vec)*ds

    # Viscous component

    mu = Constant(1.0)
    Pr = Constant(0.72)
    x = SpatialCoordinate(mesh)
    def F_v(U):
        rho, rhou1, rhou2, rhoE = U
        u1, u2, E = rhou1/rho, rhou2/rho, rhoE/rho
        u = as_vector((u1, u2))
        tau = mu*(grad(u) + grad(u).T - 2.0/3.0*(div(u))*Identity(2))
        KT = mu*gamma/Pr*(E - 0.5*(u1**2 + u2**2))
        return as_matrix([[0.0, 0.0],
                          [tau[0,0], tau[0,1]],
                          [tau[1,0], tau[1,1]],
                          [dot(tau[0,:], u) + Dx(KT, 0), (dot(tau[1,:], u)) + Dx(KT,1)]])

    viscous_domain = inner(F_v(u_vec), grad(v_vec))*dx

    C_IP = 20.0
    h = CellVolume(mesh)/FacetArea(mesh)
    sig = C_IP*Constant(max(poly_o**2, 1))/h
    G = compressible_ns_G(u_vec, mu, Pr, gamma)

    vt = DGFemViscousTerm(F_v, u_vec, v_vec, sig, G, n)
    visc_interior = vt.interior_residual(dS)
    visc_exterior = vt.exterior_residual(gD, ds)

    f = Expression(('0.8*cos(2.0*x[0] + 2.0*x[1])',
                    '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                    '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                    '(2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1]) + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5) + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4) + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3) + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2) + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1]) + 3817.472*sin(4.0*x[0] + 4.0*x[1]) + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 4)'),
                   element=V.ufl_element())

    residual = convective_domain + convective_interior + convective_exterior
    residual += viscous_domain + visc_interior + visc_exterior
    residual -= dot(f, v_vec)*dx

    J = derivative(residual, u_vec, du)
    soln_vec = Function(V)

    solve(residual == 0, u_vec, bcs=[], J=J, solver_parameters={"newton_solver": {"linear_solver": "lu"}})

    errorl2[run_count] = errornorm(gD, u_vec, norm_type='l2', degree_rise=1)
    errorh1[run_count] = errornorm(gD, u_vec, norm_type='h1', degree_rise=1)
    hsizes[run_count] = mesh.hmax()
    run_count+=1


if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])
