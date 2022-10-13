import numpy as np
import ufl

from dolfin import *
from ufl.algorithms import apply_restrictions
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_restrictions import apply_restrictions
from ufl.formatting import ufl2unicode
from ufl.formatting.ufl2unicode import ufl2unicode

from dolfin_dg import *
from dolfin_dg.dg_first_order_primal import DGDivIBP_SIPG, DGGradIBP_SIPG, PrimalHomogeneityTensor
from dolfin_dg.dg_ufl import TensorJump, Avg, apply_dg_operator_lowering, Jump, jump, avg, \
    tensor_jump, apply_average_lowering#, #apply_dg_restrictions

__author__ = 'njcs4'

# We compute the DG approximation of
#   -∇²u = f   in  Ω
#      u = gᴰ  on ∂Ω
# where Ω is the unit cube. We use a Krylov iterative method
# to approximate the solution of the underlying linear system.
# f and gᴰ are formulated for the (a priori known) solution
#      u = sin(πx)sin(πy)sin(πz).

parameters['std_out_all_processes'] = False
parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
poly_o = 2

for ele_n in ele_ns:
    mesh = UnitSquareMesh(ele_n, ele_n)
    n = FacetNormal(mesh)

    V = FunctionSpace(mesh, 'DG', poly_o)
    u, v = Function(V), TestFunction(V)
    u.rename("u", "u")

    gD = Expression('sin(pi*x[0])*sin(pi*x[1])',
                    element=V.ufl_element(), domain=mesh)
    f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])',
                   element=V.ufl_element())

    def Fv_1(u, op_u):
        return op_u

    def Fv_2(u, grad_u):
        return grad_u

    # f = -div(F_v(gD, grad(gD)))

    G1 = PrimalHomogeneityTensor(operator=lambda u: grad(u))
    penalty1 = generate_default_sipg_penalty_term(u, Constant(10.0))
    eo1 = DGGradIBP_SIPG(Fv_1, u, grad(v), penalty1, G1, n, number_ibp=2)

    G2 = PrimalHomogeneityTensor(operator=lambda sigma: div(sigma))
    penalty2 = generate_default_sipg_penalty_term(u, Constant(10.0))
    eo2 = DGDivIBP_SIPG(Fv_2, grad(u), v, penalty2, G2, n, number_ibp=1)
    eo2.temp_u = u

    F = dot(Fv_2(u, grad(Fv_1(u, u))), grad(v))*dx
    # F = dot(Fv_2(u, grad(u)), grad(v))*dx
    F += eo1.interior_residual(dS)
    F += eo1.exterior_residual(Constant(0.0), ds)
    F += eo2.interior_residual(dS)
    F += eo2.exterior_residual(Constant(0.0), ds)

    F += - f*v*dx
    J = derivative(F, u)

    info("Solving problem with %d degrees of freedom" % V.dim())
    solve(F == 0, u, J=J)#, bcs=DirichletBC(V, Constant(0.0), "on_boundary"))

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print("L2 convergence rates: " + str(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])))
    print("H1 convergence rates: " + str(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])))