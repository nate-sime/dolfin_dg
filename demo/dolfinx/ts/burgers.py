from dolfin import *
from dolfin_dg import *

parameters["std_out_all_processes"] = False
parameters["ghost_mode"] = "shared_facet"

mesh = UnitIntervalMesh(512)
V = FunctionSpace(mesh, "DG", 1)
u_soln = Expression("sin(pi*x[0] + t)",
                    degree=4, domain=mesh, t=0.0)
u = project(u_soln, V)
v = TestFunction(V)
du = TrialFunction(V)

bcs = DGDirichletBC(ds, u_soln)

def F_c(u):
    return u**2/2

g = Expression("cos(pi*x[0] + t) + 0.5*pi*sin(2.0*(pi*x[0] + t))", degree=4, t=0.0)
ho = HyperbolicOperator(mesh, V, bcs, F_c,
                        LocalLaxFriedrichs(lambda u, n: dot(u, n)))

class Burgers(PETScTSProblem):
    def F(self, u, u_t, u_tt):
        return dot(u_t, v)*dx + ho.generate_fem_formulation(u, v) - g*v*dx

PETScOptions.set("ts_monitor")
PETScOptions.set("ts_type", "rosw")
PETScOptions.set("ts_atol", 1e-3)
PETScOptions.set("ts_exact_final_time", "matchstep")

ts = PETScTSSolver(Burgers(), u, [], tdexprs=[g, u_soln])
ts.set_from_options()
ts.solve(t0=0.0, dt=1e-1, max_t=10.0, max_steps=300)

t = ts.ts.getTime()
u_soln.t = t
error_l2 = assemble((u_soln - u)**2*dx)**0.5
error_h1 = assemble(inner(grad(u_soln - u), grad(u_soln - u))**2*dx)**0.5
info("‖u(x, t) - u_h(x, t)‖L₂ = %.3e, |u(x, t) - u_h(x, t)|H¹ = %.3e"
     % (error_l2, error_h1))