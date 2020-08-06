import numpy as np
from dolfin import *

from dolfin_dg import *

parameters["ghost_mode"] = "shared_facet"
parameters['form_compiler']['representation'] = 'uflacs'

run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 2

for ele_n in ele_ns:
    # Mesh and function space.
    mesh = UnitSquareMesh(ele_n, ele_n)
    V = FunctionSpace(mesh, 'DG', p)

    # Set up Dirichlet BC
    gD = Expression('exp(x[0] - x[1])', element=V.ufl_element())
    f = Expression('(exp(x[0] - x[1]) - 1)*exp(x[0] - x[1])', element=V.ufl_element())
    u, v = interpolate(gD, V), TestFunction(V)

    bo = SpacetimeBurgersOperator(mesh, V, DGDirichletBC(ds, gD))
    residual = bo.generate_fem_formulation(u, v) - f*v*dx

    solve(residual == 0, u)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))