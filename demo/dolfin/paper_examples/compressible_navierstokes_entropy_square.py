from dolfin import *
from dolfin_dg import *
import numpy as np

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']["quadrature_degree"] = 8
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

def V_to_U(V):
    V1, V2, V3, V4 = V
    U = as_vector([-V4, V2, V3, 1 - 0.5*(V2**2 + V3**2)/V4])
    s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
    rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*exp(-s/(gamma-1))
    U = U*rhoi
    return U

poly_o = 1
run_count = 0
mesh_sizes = [8, 16, 32, 64, 128]

errorl2 = np.array([0]*len(mesh_sizes), dtype=np.double)
errorh1 = np.array([0]*len(mesh_sizes), dtype=np.double)
hsizes = np.array([0]*len(mesh_sizes), dtype=np.double)

for n_eles in mesh_sizes:
    # set up the mesh
    mesh = RectangleMesh(Point(0, 0), Point(pi, pi), n_eles, n_eles, 'right')
    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)


    # dirichlet conditions and error suite
    gamma = 1.4
    gD = Expression(('((-std::log((0.4*sin(2*x[0] + 2*x[1]) + 1.6)*pow(sin(2*x[0] + 2*x[1]) + 4, -1.4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4)) + 2.4)*(sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4) - pow(sin(2*x[0] + 2*x[1]) + 4, 2))/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '(-sin(2*x[0] + 2*x[1]) - 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))'),
                    element=V.ufl_element())

    f = Expression(('0.8*cos(2.0*x[0] + 2.0*x[1])',
                    '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                    '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                    '(2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1]) + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5) + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4) + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3) + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2) + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1]) + 3817.472*sin(4.0*x[0] + 4.0*x[1]) + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 4)'),
                   element=V.ufl_element())

    V_vec, V_test = interpolate(gD, V), TestFunction(V)

    bo = CompressibleNavierStokesOperatorEntropyFormulation(mesh, V, DGDirichletBC(ds, gD))
    residual = bo.generate_fem_formulation(V_vec, V_test) - inner(f, V_test)*dx

    solve(residual == 0, V_vec, solver_parameters={'newton_solver': {'absolute_tolerance': 5e-10}})


    gD_u = Expression(('sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                    element=V.ufl_element())

    u_soln = project(V_to_U(V_vec), V)
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
