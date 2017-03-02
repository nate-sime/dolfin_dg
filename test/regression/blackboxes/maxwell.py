from dolfin import *
import numpy as np
from dolfin_dg.operators import MaxwellAndInvolutionOperator, DGDirichletBC

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
# parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [1, 2, 4, 8]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
ell = 1

for ele_n in ele_ns:
    mesh = UnitCubeMesh(ele_n, ele_n, ele_n)

    V = FunctionSpace(mesh, MixedElement([VectorElement('DG', tetrahedron, ell), FiniteElement('DG', tetrahedron, ell+1)]))
    u, v = Function(V), TestFunction(V)
    E, p = split(u)
    F, q = split(v)

    gD = Expression(['sin(pi*x[0])*sin(pi*x[1])']*3, degree=ell+1)
    f = Expression(("pow(pi, 2)*(sin(pi*(-x[0] + x[1] + x[2])) + 0.5*sin(pi*(x[0] - x[1] + x[2])) + 0.5*sin(pi*(x[0] + x[1] - x[2])))",
                    "pow(pi, 2)*(0.5*sin(pi*(-x[0] + x[1] + x[2])) + sin(pi*(x[0] - x[1] + x[2])) + 0.5*sin(pi*(x[0] + x[1] - x[2])))",
                    "pow(pi, 2)*(0.5*sin(pi*(-x[0] + x[1] + x[2])) + 0.5*sin(pi*(x[0] - x[1] + x[2])) + sin(pi*(x[0] + x[1] - x[2])))"),
                    degree=ell+1)

    pe = MaxwellAndInvolutionOperator(mesh, V, DGDirichletBC(ds, gD))
    residual = pe.generate_fem_formulation(u, v) - inner(f, F)*dx

    solve(residual == 0, u, [], solver_parameters={"newton_solver": {"linear_solver": "mumps", "maximum_iterations": 2, "error_on_nonconvergence": False}})
    plot(u.sub(0))
    interactive()

    errorl2[run_count] = assemble(inner(gD - E, gD - E)*dx)**0.5 #errornorm(gD, u.sub(0), norm_type='l2', degree_rise=3)
    # errorh1[run_count] = errornorm(gD, u.sub(0), norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])


