from firedrake import *
from dolfin_dg import *
import numpy as np


run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

dx = dx(degree=8)
dS = dS(degree=8)
ds = ds(degree=8)

for ele_n in ele_ns:
    # Mesh and function space.
    mesh = RectangleMesh(ele_n, ele_n, pi, pi, diagonal='right')
    V = VectorFunctionSpace(mesh, 'DG', p, dim=4)

    # Set up Dirichlet BC
    x = SpatialCoordinate(mesh)
    gD = as_vector([sin(2*(x[0]+x[1])) + 4,
                    0.2*sin(2*(x[0]+x[1])) + 4,
                    0.2*sin(2*(x[0]+x[1])) + 4,
                    pow((sin(2*(x[0]+x[1])) + 4), 2)])

    f = as_vector([0.8*cos(2.0*x[0] + 2.0*x[1]),
                   (1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3),
                   (1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3),
                   (2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1]) + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5) + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4) + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3) + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2) + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1]) + 3817.472*sin(4.0*x[0] + 4.0*x[1]) + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 4)])

    u, v = project(gD, V), TestFunction(V)

    bo = CompressibleNavierStokesOperator(mesh, V, DGDirichletBC(ds, gD))
    residual = bo.generate_fem_formulation(u, v, dx=dx, dS=dS) - inner(f, v)*dx

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
    hsizes[run_count] = pi/ele_n

    run_count += 1

print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
