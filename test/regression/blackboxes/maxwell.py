from dolfin import *
import numpy as np

from dolfin_dg.dg_form import DGFemCurlTerm
from dolfin_dg.operators import DGDirichletBC

__author__ = 'njcs4'

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

    V = VectorFunctionSpace(mesh, "DG", 1)
    v = TestFunction(V)

    k = Constant(2.0)
    gD = Expression(("sin(k*x[1])", "sin(k*x[0])"), k=k, degree=ell+1)
    u = interpolate(gD, V)

    def F_m(u, curl_u):
        return curl_u

    curl_u = variable(curl(u))
    tau = F_m(u, curl_u)

    G = diff(tau, curl_u)

    n = FacetNormal(mesh)
    h = CellVolume(mesh)/FacetArea(mesh)
    gamma = 20.0
    sigma = Constant(gamma*max(ell**2, 1))/h

    mt = DGFemCurlTerm(F_m, u, v, sigma, G, n)
    residual = dot(curl(u), curl(v))*dx - k**2*dot(u, v)*dx
    residual += mt.interior_residual(dS)
    residual += mt.exterior_residual(gD, ds)

    solve(residual == 0, u, [], solver_parameters={"newton_solver": {"linear_solver": "mumps"}})
    plot(u)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='Hcurl', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])

# interactive()