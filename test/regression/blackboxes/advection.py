from dolfin import *
import numpy as np
from dolfin_dg.dg_form import DGFemViscousTerm
from dolfin_dg.fluxes import lax_friedrichs_flux
from dolfin_dg.operators import HyperbolicOperator, DGDirichletBC

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
p = 2

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n, 'left/right')

    V = FunctionSpace(mesh, 'DG', p)
    v = TestFunction(V)

    gD = Expression('exp(x[0] - x[1])', element=V.ufl_element())
    u = interpolate(gD, V)
    f = Expression('0.0',
                   element=V.ufl_element())
    b = Constant((1, 1))
    n = FacetNormal(mesh)

    # Convective Operator
    def F_c(U):
        return b*U**2

    ho = HyperbolicOperator(mesh, V, DGDirichletBC(ds, gD), F_c, alpha=lambda u, n: 2*u*dot(b, n))
    residual = ho.generate_fem_formulation(u, v) - f*v*dx

    du = TrialFunction(V)
    J = derivative(residual, u, du)
    solve(residual == 0, u, [], J=J)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])


