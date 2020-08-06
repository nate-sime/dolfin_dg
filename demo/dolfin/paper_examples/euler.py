import numpy as np
from dolfin import *

from dolfin_dg import *

parameters["ghost_mode"] = "shared_facet"
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']["quadrature_degree"] = 8

run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

for ele_n in ele_ns:
    # Mesh and function space.
    mesh = RectangleMesh(Point(0, 0), Point(.5*pi, .5*pi), ele_n, ele_n, 'right')
    V = VectorFunctionSpace(mesh, 'DG', p, dim=4)

    # Set up Dirichlet BC
    gD = Expression(('sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     '0.2*sin(2*(x[0]+x[1])) + 4',
                     'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                    element=V.ufl_element())

    f = Expression(('(4.0L/5.0L)*cos(2*x[0] + 2*x[1])',
                    '(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)',
                    '(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)',
                    '(8.0L/625.0L)*(175*pow(sin(2*x[0] + 2*x[1]), 4) + 4199*pow(sin(2*x[0] + 2*x[1]), 3) + 33588*pow(sin(2*x[0] + 2*x[1]), 2) + 112720*sin(2*x[0] + 2*x[1]) + 145600)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 3)'),
                   element=V.ufl_element())

    u, v = interpolate(gD, V), TestFunction(V)

    bo = CompressibleEulerOperator(mesh, V, DGDirichletBC(ds, gD))
    residual = bo.generate_fem_formulation(u, v) - inner(f, v)*dx

    solve(residual == 0, u)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))