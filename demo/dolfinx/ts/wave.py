from dolfin import *
from dolfin_dg import *

if MPI.size(MPI.comm_world) > 1:
    raise NotImplementedError("Demo designed to run in serial only.")

mesh = UnitIntervalMesh(128)
V = FunctionSpace(mesh, "CG", 1)
u, v = Function(V), TestFunction(V)
du = TrialFunction(V)

bcs = DirichletBC(V, Constant(0.0), "on_boundary")

class WaveProblem(PETScTSProblem):
    def F(self, u, u_t, u_tt):
        return u_tt*v*dx + dot(grad(u), grad(v))*dx

PETScOptions.set("ts_monitor")
PETScOptions.set("ts_type", "alpha2")
PETScOptions.set("ts_exact_final_time", "matchstep")

import matplotlib.pyplot as plt

u_soln = Expression("sin(pi*x[0])*cos(pi*t)", t=0, degree=4)
project(u_soln, V, function=u)

ts = PETScTSSolver(WaveProblem(), u, bcs)
ts.set_from_options()
ts.solve(t0=0.0, dt=1e-2, max_t=10.0, max_steps=1001,
         scheme=PETScTSScheme.implicit2)

t = ts.ts.getTime()
u_soln.t = t
error_l2 = assemble((u_soln - u)**2*dx)**0.5
u_eval = u(0.5)
error_point = abs(1.0 - u_eval/float(exp(-pi**2*t)))

print("‖u(x, t) - u_h(x, t)‖L₂ = %.3e, |1 - u_h(0.5, t)/u(0.5, t)|H¹ = %.3e"
      % (error_l2, error_point))

plot(u)
plt.show()