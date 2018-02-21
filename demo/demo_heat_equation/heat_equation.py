import ufl
import matplotlib.pyplot as plt
import numpy as np

from dolfin import *
from dolfin_dg import *

# Here we solve the problem: find u in the unit interval which satisfies
#   u_t = u_xx, u(0, t) = u(1, t) = 0, u(x, 0) = sin(pi*x),
# which has known solution
#   u = sin(pi*x)*exp(-pi^2*t).

mesh = UnitIntervalMesh(32)
V = FunctionSpace(mesh, "DG", 1)
u, v = Function(V), TestFunction(V)
un = Function(V)
du = TrialFunction(V)

# Define a theta scheme using Crank-Nicholson correction.
dt = Constant(1e-2)
theta = Constant(0.5)
uth = theta*u + (1 - theta)*un

# Construct the DG discretisation of the spatial derivative of the
# heat equation using the Poisson operator. We then replace u with the
# theta scheme variable uth.
pe = PoissonOperator(mesh, V, DGDirichletBC(ds, Constant(0.0)))
a_term = ufl.replace(pe.generate_fem_formulation(u, v), {u: uth})

# Newton's method applied to the semilinear residual is equivalent
# to solving the linear problem a(u, v) = l(v)
F = (u - un)*v*dx + dt*(a_term - Constant(0.0)*v*dx)
a = derivative(F, u, du)
L = -F

# Project the ininital condition onto the FE solution space.
# This is the best approximation in the FE space, (cf. Cea's Lemma).
# Compare this, for example, with interpolating the initial condition.
u0 = Expression("sin(pi*x[0])", degree=4)
project(u0, V, function=un)
# un.interpolate(u0)

# Analytical solution
uex = Expression("sin(pi*x[0])*exp(-pi*pi*t)", t=0.0, degree=4)

# Store the computed l2 error at each time step
errors = [errornorm(uex, un, "l2")]

# Time evolution parameters
tmax = 0.1
maxsteps = 1000
t = 0.0

# Begin the iteration over time steps
for j in range(maxsteps):
    # Update time
    t = float(dt)*(j + 1)
    uex.t = t

    # If we've advanced beyond the time interval, stop.
    if float(t) > tmax:
        break

    # Solve the system at the current time step and store the L2 error
    solve(a == L, un)
    errors.append(errornorm(uex, un, "l2"))

# Display error metrics
info("Steps taken: %d, L2 error at final time step: %.3e" % (j + 1, errors[-1]))
plt.plot(np.arange(0.0, tmax+float(dt), float(dt)), errors)
plt.xlabel(r"$t$")
plt.ylabel(r"$\Vert u(x, t) - u_h(x, t)\Vert_{L_2(\Omega)}$")
plt.xlim((0, tmax))
plt.show()