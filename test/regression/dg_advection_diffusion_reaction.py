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
p = 2

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n, 'left/right')

    V = FunctionSpace(mesh, 'DG', p)
    v = TestFunction(V)

    gD = Expression('exp(x[0] - x[1])', element=V.ufl_element())
    u = interpolate(gD, V)
    f = Expression('-4*exp(2*(x[0] - x[1])) - 2*exp(x[0] - x[1])',
                   element=V.ufl_element())

    n = FacetNormal(mesh)
    h = CellVolume(mesh)/FacetArea(mesh)
    A = u + 1
    b = as_vector((1, 1))

    # Poisson homogeneity tensor
    G = {}
    G[0,0] = as_matrix(((A,),))
    G[0,1] = as_matrix(((0,),))
    G[1,0] = as_matrix(((0,),))
    G[1,1] = as_matrix(((A,),))

    # Convective Operator
    def F_c(U):
        return b*U**2

    conv_volume = -inner(F_c(u), grad(v))*dx

    H = LocalLaxFriedrichs(lambda u, n: 2*u*dot(b, n))
    H.setup(F_c, u("+"), u("-"), n("+"))
    conv_interior = dot(H.interior(F_c, u('+'), u('-'), n('+')), (v('+') - v('-')))*dS

    H.setup(F_c, u, gD, n)
    conv_exterior = dot(H.exterior(F_c, u, gD, n), v)*ds

    # Viscous Operator
    def F_v(u):
        return A*grad(u)

    sig = Constant(20.0)*max(p**2, 1)/h
    vt = DGFemSIPG(F_v, u, v, sig, G, n)
    visc_volume = inner(F_v(u), grad(v))*dx
    visc_interior = vt.interior_residual(dS)
    visc_exterior = vt.exterior_residual(gD, ds)

    residual = conv_volume + conv_interior + conv_exterior \
             + visc_volume + visc_interior + visc_exterior \
             - f*v*dx

    du = TrialFunction(V)
    J = derivative(residual, u, du)
    solve(residual == 0, u, J=J)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))


