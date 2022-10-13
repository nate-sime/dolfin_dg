import dolfinx
import dolfin_dg
import dolfin_dg.dolfinx.ts
import ufl
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc


if MPI.COMM_WORLD.size > 1:
    raise NotImplementedError("Demo designed to run in serial only.")

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 128)
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = dolfinx.fem.Function(V), ufl.TestFunction(V)
du = ufl.TrialFunction(V)

bc_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: np.full_like(x[0], True, dtype=np.int8))
bc_dofs = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim-1, bc_facets)
bcs = dolfinx.fem.dirichletbc(
    np.array(0.0, dtype=np.double), bc_dofs, V)

class HeatProblem(dolfin_dg.dolfinx.ts.PETScTSProblem):
    def F(self, u, u_t, u_tt):
        return u_t * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

# PETScOptions.set("ts_monitor")
# PETScOptions.set("ts_type", "rosw")
# PETScOptions.set("ts_atol", 1e-6)
# PETScOptions.set("ts_exact_final_time", "matchstep")
t = 0.0
u.interpolate(lambda x: np.sin(np.pi*x[0])*np.exp(-np.pi**2*t))
ts = dolfin_dg.dolfinx.ts.PETScTSSolver(HeatProblem(), u, bcs)

opts = PETSc.Options()
opts["ts_monitor"] = None
opts["ts_type"] = "rosw"
opts["ts_atol"] = 1e-6
opts["ts_exact_final_time"] = "matchstep"
ts.set_from_options()
ts.solve(t0=0.0, dt=1e-2, max_t=1.0, max_steps=1001)

import matplotlib.pyplot as plt

t = ts.ts.getTime()
u_soln.t = t
error_l2 = assemble((u_soln - u)**2*dx)**0.5
u_eval = u(0.5)
error_point = abs(1.0 - u_eval/float(exp(-pi**2*t)))

print("‖u(x, t) - u_h(x, t)‖L₂ = %.3e, |1 - u_h(0.5, t)/u(0.5, t)|H¹ = %.3e"
      % (error_l2, error_point))

plot(u)
plt.show()