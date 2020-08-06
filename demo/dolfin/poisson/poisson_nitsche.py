import numpy as np
from dolfin import (
    UnitSquareMesh, FunctionSpace, Function, TestFunction, dx, errornorm,
    MPI, Expression, grad, dot, div, solve, parameters, ds)

from dolfin_dg import NitscheBoundary

# We compute the DG approximation of
#   -∇·(u+1)∇u = f   in  Ω
#      u = gᴰ  on ∂Ω
# where Ω is the unit cube. We use Nitsche's method to impose
# the boundary condition. f and gᴰ are formulated for the
# (a priori known) solution
#      u = sin(πx)sin(πy).

parameters["ghost_mode"] = "shared_facet"

p_order = 1
run_count = 0
ele_ns = [4, 8, 16, 32, 64, 128]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))

for j, ele_n in enumerate(ele_ns):
    mesh = UnitSquareMesh(ele_n, ele_n)

    V = FunctionSpace(mesh, 'CG', p_order)
    u, v = Function(V), TestFunction(V)

    gD = Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0',
                    domain=mesh, degree=p_order+1)

    def F_v(u, grad_u):
        return (u + 1) * grad_u
    bc = NitscheBoundary(F_v, u, v)

    f = -div(F_v(gD, grad(gD)))
    residual = dot(F_v(u, grad(u)), grad(v)) * dx - f * v * dx
    residual += bc.nitsche_bc_residual(gD, ds)

    solve(residual == 0, u)

    errorl2[j] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[j] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[j] = mesh.hmax()


if MPI.rank(mesh.mpi_comm()) == 0:
    hrates = np.log(hsizes[0:-1] / hsizes[1:])
    print("L2 convergence rates: "
          + str(np.log(errorl2[0:-1] / errorl2[1:]) / hrates))
    print("H1 convergence rates: "
          + str(np.log(errorh1[0:-1] / errorh1[1:]) / hrates))
