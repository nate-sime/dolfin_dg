import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from dolfin_dg import *

__author__ = 'njcs4'

parameters["ghost_mode"] = "shared_facet"

if MPI.size(MPI.comm_world) > 1:
    raise NotImplementedError("Demo does not currently work in parallel")


class EndTimeInterpolation(UserExpression):

    def set_u_and_dt(self, u, dt):
        self.u = u
        self.dt = dt

    def eval(self, value, x):
        value[0] = self.u(Point(x[0], self.dt))


# The size of the spatial domain and the time step
a = 1.0
t_max = 0.2
t_steps = 64
dt = t_max/float(t_steps)

# The mesh extrudes one element into time
mesh = RectangleMesh(Point(0.0, 0.0), Point(a, dt), 64, 1)
V = FunctionSpace(mesh, 'DG', 1)
du = TrialFunction(V)

# Set up initial condition
u0 = Expression('max(sin(2.0*pi*x[0]/a), 0.0)',
                a=a, element=V.ufl_element())
u = project(u0, V)

# Use a custom Expression to interpolate the computed u(x,t+dt)
# as the initial condition at the next time step.
u_dt = EndTimeInterpolation(element=V.ufl_element())
u_dt.set_u_and_dt(project(u0, V), dt)
v = TestFunction(V)

# Establish exterior boundary components and construct measure
exterior_bdries = MeshFunction('size_t', mesh,
                               mesh.topology().dim()-1, 0)
CompiledSubDomain("on_boundary").mark(exterior_bdries, 2)
CompiledSubDomain("near(x[1], 0.0)").mark(exterior_bdries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=exterior_bdries)

bcs = [DGDirichletBC(ds(1), u_dt), DGNeumannBC(ds(2), u)]

# Automatic formulation of the DG spacetime Burgers problem
llf = LocalLaxFriedrichs(lambda u, n: u*n[0] + n[1])
bo = SpacetimeBurgersOperator(mesh, V, bcs, flux=llf)
residual = bo.generate_fem_formulation(u, v)
J = derivative(residual, u, du)

# Time loop
t = 0
for j in range(t_steps):
    t += dt
    solve(residual == 0, u)
    u_dt.u.vector()[:] = u.vector()[:]

# Interpolate the solution at the final time for plotting
x = np.linspace(0, 1, 100)
y = np.array(list(map(lambda x: u(x, dt), x)))

plt.xlabel("$x$")
plt.ylabel("$u_h(x,t=%.1f)$" % t)
plt.plot(x, y)
plt.show()