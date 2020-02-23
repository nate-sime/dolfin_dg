from firedrake import *
from dolfin_dg import *
import numpy as np


run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1
quad_degree = 8

for ele_n in ele_ns:
    # Mesh and function space.
    mesh = RectangleMesh(ele_n, ele_n, pi/2, pi/2, diagonal="right")
    V = VectorFunctionSpace(mesh, 'DG', p, dim=4)

    dx = dx(degree=quad_degree)
    ds = ds(degree=quad_degree)
    dS = dS(degree=quad_degree)

    # Set up Dirichlet BC
    x = SpatialCoordinate(mesh)
    gD = as_vector((sin(2*(x[0]+x[1])) + 4,
                    0.2*sin(2*(x[0]+x[1])) + 4,
                    0.2*sin(2*(x[0]+x[1])) + 4,
                    pow((sin(2*(x[0]+x[1])) + 4), 2)))

    f = as_vector(((4.0/5.0)*cos(2*x[0] + 2*x[1]),
                    (8.0/125.0)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2),
                    (8.0/125.0)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2),
                    (8.0/625.0)*(175*pow(sin(2*x[0] + 2*x[1]), 4) + 4199*pow(sin(2*x[0] + 2*x[1]), 3) + 33588*pow(sin(2*x[0] + 2*x[1]), 2) + 112720*sin(2*x[0] + 2*x[1]) + 145600)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 3)))

    u, v = project(gD, V), TestFunction(V)

    bo = CompressibleEulerOperator(mesh, V, DGDirichletBC(ds, gD))
    residual = bo.generate_fem_formulation(u, v) - inner(f, v)*dx

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
    hsizes[run_count] = (pi/2) / ele_n

    run_count += 1

print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
