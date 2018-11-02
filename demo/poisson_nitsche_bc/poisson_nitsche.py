from dolfin import *
from dolfin_dg import *
import numpy as np

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32, 64, 128]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n, 'left')

    V = FunctionSpace(mesh, 'CG', p)
    u, v = Function(V), TestFunction(V)

    gD = Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0', element=V.ufl_element())
    f = Expression('2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)*sin(pi*x[0])*sin(pi*x[1]) - pow(pi, 2)*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2) - pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)',
                   element=V.ufl_element())

    u.interpolate(Constant(0.0))
    # pe = PoissonOperator(mesh, V, DGDirichletBC(ds, gD), kappa=u + 1)
    # residual = pe.generate_fem_formulation(u, v) - f*v*dx

    F_v = lambda u, grad_u: (u+1)*grad_u
    G = homogeneity_tensor(F_v, u)
    h = CellDiameter(mesh)#CellVolume(mesh)/FacetArea(mesh)
    C_IP = Constant(20.0)
    n = FacetNormal(mesh)
    sigma = C_IP*Constant(max(V.ufl_element().degree()**2, 1))/h
    vt = DGFemSIPG(F_v, u, v, sigma, G, n)

    residual = dot(F_v(u, grad(u)), grad(v))*dx - f*v*dx
    residual += vt.exterior_residual(gD, ds) #+ vt.interior_residual(dS)


    bcs = []# DirichletBC(V, gD, "on_boundary")

    solve(residual == 0, u, bcs, solver_parameters={"newton_solver": {"linear_solver": "lu"}})

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print(errorl2)
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))


