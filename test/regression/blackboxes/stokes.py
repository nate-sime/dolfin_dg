from dolfin import *
import numpy as np
from dolfin_dg.operators import StokesEquations, DGDirichletBC

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
ell = 1

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n, 'left/right')

    V = FunctionSpace(mesh, MixedElement([VectorElement("DG", triangle, ell), FiniteElement("DG", triangle, ell-1)]))
    u_vec, v_vec = Function(V), TestFunction(V)

    u, p = split(u_vec)
    v, q = split(v_vec)

    gD = Expression(('-(x[1]*cos(x[1]) + sin(x[1]))*exp(x[0])', 'x[1]*sin(x[1])*exp(x[0])'), degree=(ell+1))
    pD = Expression('2.0*exp(x[0])*sin(x[1])-1.5797803888225995912/3.0', degree=ell+1)
    f = Constant((0.0, 0.0))

    pe = StokesEquations(mesh, V, DGDirichletBC(ds, gD))
    print pe.generate_fem_formulation(u_vec, v_vec)
    quit()
    residual = pe.generate_fem_formulation(u_vec, v_vec) - inner(f, v)*dx

    solve(residual == 0, u, [], solver_parameters={"newton_solver": {"linear_solver": "lu"}})

    u, _ = split(u_vec)
    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])


