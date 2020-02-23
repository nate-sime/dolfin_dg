from firedrake import *
from dolfin_dg import *
import numpy as np


run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 2

for ele_n in ele_ns:
    # Mesh and function space.
    mesh = UnitSquareMesh(ele_n, ele_n)
    V = FunctionSpace(mesh, 'DG', p)

    # Set up Dirichlet BC
    x = SpatialCoordinate(mesh)
    gD = exp(x[0] - x[1])
    f = (exp(x[0] - x[1]) - 1)*exp(x[0] - x[1])
    u, v = project(gD, V), TestFunction(V)

    bo = SpacetimeBurgersOperator(mesh, V, DGDirichletBC(ds, gD))
    residual = bo.generate_fem_formulation(u, v) - f*v*dx

    sp = {
          "snes_type": "newtonls",
          "snes_monitor": None,
          "snes_linesearch_type": "basic",
          "snes_linesearch_maxstep": 1,
          "snes_linesearch_damping": 1,
          "snes_linesearch_monitor": None,
          "snes_max_it": 10000,
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"
         }
    solve(residual == 0, u, solver_parameters=sp)

    errorl2[run_count] = errornorm(gD, u, norm_type='L2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='H1', degree_rise=3)
    hsizes[run_count] = 1/ele_n

    run_count += 1

print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
