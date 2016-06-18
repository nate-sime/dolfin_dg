from dolfin import *
import numpy as np
from dolfin_dg.dg_form import DGFemViscousTerm

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n, 'left/right')

    V = FunctionSpace(mesh, 'DG', p)
    u = Function(V)
    v = TestFunction(V)

    gD = Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0', element=V.ufl_element())

    n = FacetNormal(mesh)
    h = CellVolume(mesh)/FacetArea(mesh)
    gamma = 20.0
    sig = gamma*max(p**2, 1)/h

    def F_v(u):
        return (u+1)*grad(u)

    G = {}
    G[0,0] = as_matrix(((u + 1,),))
    G[0,1] = as_matrix(((0,),))
    G[1,0] = as_matrix(((0,),))
    G[1,1] = as_matrix(((u + 1,),))

    vt = DGFemViscousTerm(F_v, u, v, sig, G, n)
    interior = vt.interior_residual(dS)
    exterior = vt.exterior_residual(gD, ds)

    f = Expression('2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)*sin(pi*x[0])*sin(pi*x[1]) - pow(pi, 2)*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2) - pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)',
                   element=V.ufl_element())
    residual = inner((u+1)*grad(u), grad(v))*dx + interior + exterior - f*v*dx

    solve(residual == 0, u, [], solver_parameters={"newton_solver": {"linear_solver": "lu"}})

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])


