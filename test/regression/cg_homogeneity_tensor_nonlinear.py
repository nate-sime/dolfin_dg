from dolfin import *
from dolfin_dg import *
import numpy as np

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n, 'left/right')

    V = VectorFunctionSpace(mesh, 'CG', p, dim=4)
    u = Function(V)
    v = TestFunction(V)

    gD = Expression(['sin(pi*x[0])*sin(pi*x[1]) + 1.0']*u.ufl_shape[0], element=V.ufl_element())
    f = Expression(['2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)*sin(pi*x[0])*sin(pi*x[1]) - pow(pi, 2)*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2) - pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)']*u.ufl_shape[0],
                   element=V.ufl_element())

    def F_v(u, grad_u):
        return as_vector([(u[j] + 1)*grad_u[j, :] for j in range(u.ufl_shape[0])])

    G = homogeneity_tensor(F_v, u)
    residual = inner(hyper_tensor_product(G, grad(u)), grad(v))*dx - dot(f, v)*dx


    bcs = DirichletBC(V, gD, 'on_boundary')
    solve(residual == 0, u, bcs)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))


