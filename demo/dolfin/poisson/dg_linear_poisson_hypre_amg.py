import numpy as np

from dolfin import *
from dolfin_dg import *

__author__ = 'njcs4'

# We compute the DG approximation of
#   -∇²u = f   in  Ω
#      u = gᴰ  on ∂Ω
# where Ω is the unit cube. We use a Krylov iterative method
# to approximate the solution of the underlying linear system.
# f and gᴰ are formulated for the (a priori known) solution
#      u = sin(πx)sin(πy)sin(πz).

parameters['std_out_all_processes'] = False
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

for ele_n in ele_ns:
    mesh = UnitCubeMesh(ele_n, ele_n, ele_n)

    V = FunctionSpace(mesh, 'DG', p)
    u, v = Function(V), TestFunction(V)

    gD = Expression('sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',
                   element=V.ufl_element())
    f = Expression('3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',
                   element=V.ufl_element())

    pe = PoissonOperator(mesh, V, DGDirichletBC(ds, gD))
    F = pe.generate_fem_formulation(u, v) - f*v*dx
    J = derivative(F, u)

    A, b = PETScMatrix(), PETScVector()
    assemble_system(J, -F, A_tensor=A, b_tensor=b)

    solver = PETScKrylovSolver()
    solver.set_operator(A)

    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("ksp_rtol", 1e-6)
    solver.set_from_options()

    info("Solving problem with %d degrees of freedom" % V.dim())
    solver.solve(u.vector(), b)

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print("L2 convergence rates: " + str(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])))
    print("H1 convergence rates: " + str(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])))