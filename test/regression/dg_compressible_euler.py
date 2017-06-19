from dolfin import *
from dolfin_dg import *
import numpy as np

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

poly_o = 1
run_count = 0
mesh_sizes = [4, 8, 16, 32, 64]

errorl2 = np.array([0]*len(mesh_sizes), dtype=np.double)
errorh1 = np.array([0]*len(mesh_sizes), dtype=np.double)
hsizes = np.array([0]*len(mesh_sizes), dtype=np.double)

for n_eles in mesh_sizes:
    mesh = RectangleMesh(Point(0, 0), Point(.5*pi, .5*pi), n_eles, n_eles, 'right')
    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
    du = TrialFunction(V)
    v_vec = TestFunction(V)

    gamma = 1.4
    gD = Expression(('sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                    element=V.ufl_element())

    u_vec = interpolate(gD, V)
    mesh_n = FacetNormal(mesh)

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
        lambdas = list(map(abs, lambdas))
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
        value = Constant(0.5)*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + alpha*(U_p - U_m))
        return value

    convective_interior = dot(H(u_vec('+'), u_vec('-'), mesh_n('+')), (v_vec('+') - v_vec('-')))*dS
    convective_exterior = dot(H(u_vec, gD, mesh_n), v_vec)*ds

    f = Expression(('(4.0L/5.0L)*cos(2*x[0] + 2*x[1])',
                    '(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)',
                    '(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)',
                    '(8.0L/625.0L)*(175*pow(sin(2*x[0] + 2*x[1]), 4) + 4199*pow(sin(2*x[0] + 2*x[1]), 3) + 33588*pow(sin(2*x[0] + 2*x[1]), 2) + 112720*sin(2*x[0] + 2*x[1]) + 145600)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 3)'),
                   element=V.ufl_element())

    residual = convective_domain + convective_interior + convective_exterior - dot(f, v_vec)*dx

    J = derivative(residual, u_vec, du)
    solve(residual == 0, u_vec, [], J=J)

    errorl2[run_count] = errornorm(gD, u_vec, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u_vec, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()
    run_count += 1


if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
