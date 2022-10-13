import matplotlib.pyplot as plt

from dolfin import *
from dolfin_dg import *

__author__ = 'njcs4'

# Reproduction of the renowned numerical experiment showcased in:
# Sod, G.A., A Survey of Several Finite Difference Methods for Systems
# of Nonlinear Hyperbolic Conservation Laws. Journal of Computational
# Physics, 1978, 27 (1).

parameters["ghost_mode"] = "shared_facet"
parameters['form_compiler']['representation'] = 'uflacs'

# Mesh and function space.
mesh = UnitIntervalMesh(1024)
V = VectorFunctionSpace(mesh, 'DG', 1, dim=3)
du = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(mesh)

# Set up the initial conditions in regions 1 and 5, left and right
# of the diaphragm, respectively.
gamma = 1.4
rho1, rho5 = 1.0, 0.125
p1, p5 = 1.0, 0.1
u1, u5 = 0.0, 0.0
e1, e5 = aero.energy_density(p1, rho1, u1), aero.energy_density(p5, rho5, u5)

# Project the initial conditions onto the left and right of the
# diaphragm by constructing a new integration measure
cf = MeshFunction("size_t", mesh, 1, 5)
CompiledSubDomain("x[0] <= 0.5").mark(cf, 1)
dx_p = Measure("dx", subdomain_data=cf)
gD1 = as_vector((rho1, rho1*u1, e1))
gD5 = as_vector((rho5, rho5*u5, e5))
u_vec = Function(V)
solve(dot(du, v)*dx == dot(gD1, v)*dx_p(1) + dot(gD5, v)*dx_p(5), u_vec)

# Generate the DGFEM formulation for the Compressible Euler operator,
# then replace the solution variable with the theta scheme variable.
eo = CompressibleEulerOperator(mesh, V, bcs=DGNeumannBC(ds, Constant(0.0)))

class ShockTube(PETScTSProblem):
    def F(self, u, u_t, u_tt):
        return dot(v, u_t)*dx + eo.generate_fem_formulation(u, v)

PETScOptions.set("ts_monitor")
PETScOptions.set("ts_type", "rosw")
PETScOptions.set("ts_atol", 1e-3)
PETScOptions.set("ts_exact_final_time", "matchstep")

ts = PETScTSSolver(ShockTube(), u_vec, [])
ts.set_from_options()
ts.solve(t0=0.0, dt=5e-4, max_t=0.2, max_steps=1000)

# Plot the density, velocity, pressure and energy density
rho, rhou, e = u_vec
plot(u_vec.sub(0))
plot(rhou/rho)
plot(aero.pressure(u_vec))
plt.legend((r"$\rho$", r"$u$", r"$p$"))
plt.xlabel(r"$x$")
plt.ylim((-0.05, 1.2))
plt.xlim((0, 1))
plt.show()
plot(e/rho)
plt.legend((r"$E$",))
plt.show()