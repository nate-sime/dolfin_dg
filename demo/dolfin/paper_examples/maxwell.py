from dolfin import *
from dolfin_dg import *
import numpy as np


parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
ell = 1

for ele_n in ele_ns:
    mesh = RectangleMesh(Point(-1., -1.), Point(1., 1.), ele_n, ele_n)

    V = VectorFunctionSpace(mesh, "DG", ell)
    v = TestFunction(V)

    k = Constant(2.0)
    gD = Expression(("sin(k*x[1])", "sin(k*x[0])"), k=k, degree=ell+1, domain=mesh)
    u = interpolate(gD, V)

    def F_m(u, curl_u):
        return curl_u

    mo = MaxwellOperator(mesh, V, [DGDirichletBC(ds, gD)], F_m)
    residual = mo.generate_fem_formulation(u, v) - k**2*dot(u, v)*dx

    solve(residual == 0, u)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='Hcurl', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))