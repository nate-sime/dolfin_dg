import ufl
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
# diaphragm by constructing a new integraiton measure
cf = MeshFunction("size_t", mesh, 1, 5)
CompiledSubDomain("x[0] <= 0.5").mark(cf, 1)
dx_p = Measure("dx", subdomain_data=cf)
gD1 = as_vector((rho1, rho1*u1, e1))
gD5 = as_vector((rho5, rho5*u5, e5))
u_vec = Function(V)
solve(dot(du, v)*dx == dot(gD1, v)*dx_p(1) + dot(gD5, v)*dx_p(5), u_vec)

# The previous time step and predictor-corrector theta scheme terms
dt = Constant(5e-4)
un = Function(V)
theta = Constant(0.5)
uth = theta*u_vec + (1.0 - theta)*un

# Generate the DGFEM formulation for the Compressible Euler operator,
# then replace the solution variable with the theta scheme variable.
eo = CompressibleEulerOperator(mesh, V, bcs=DGNeumannBC(ds, Constant(0.0)))
residual = dt*eo.generate_fem_formulation(u_vec, v)
residual = ufl.replace(residual, {u_vec: uth})
residual += dot(u_vec - un, v)*dx
J = derivative(residual, u_vec, du)


class ShockTubeProblem(NonlinearProblem):
    def F(self, b, x):
        assemble(residual, tensor=b)
    def J(self, A, x):
        assemble(J, tensor=A)


solver = PETScSNESSolver('newtonls')

# Output the components of the solution vector (rho, rho*u, e)
# to XDMF file
xdmf = XDMFFile("output.xdmf")
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.parameters["functions_share_mesh"] = True

# Run the simulation until elapsed time = 0.2
for j in range(400):
    un.assign(u_vec)
    solver.solve(ShockTubeProblem(), u_vec.vector())
    if j % 10 == 0:
        xdmf.write(u_vec.sub(0), float(dt)*(j + 1))
        xdmf.write(u_vec.sub(1), float(dt)*(j + 1))
        xdmf.write(u_vec.sub(2), float(dt)*(j + 1))

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