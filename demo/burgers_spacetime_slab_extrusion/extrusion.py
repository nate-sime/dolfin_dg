import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from dolfin_dg import *

__author__ = 'njcs4'

parameters["ghost_mode"] = "shared_facet"
parameters['form_compiler']['representation'] = 'uflacs'

class LastFunc(UserExpression):

    def set_fs(self, u, d):
        self.u = u
        self.d = d

    def eval(self, value, x):
        value[0] = self.u(Point(x[0], self.d))

# a = max distance
# d = max time
d_comp = 0.2
d_comp_n = 64
dt = d_comp/d_comp_n
a, d = 1.0, dt

class FixedBC(SubDomain):
    def inside(self, x, on):
        return near(x[1], 0.0) and on

# Mesh and function space.
mesh = RectangleMesh(Point(0.0, 0.0), Point(a, d), 64, 1)
V = FunctionSpace(mesh, 'DG', 1)
du = TrialFunction(V)

# Set up Dirichlet BC
gD = Expression('max(sin(2.0*pi*x[0]/a), 0.0)',
                a=a, element=V.ufl_element())
u = project(gD, V)
gD = project(gD, V)
gD_func = LastFunc(element=V.ufl_element())
gD_func.set_fs(gD, d)
v = TestFunction(V)

# Establish exterior boundary components and construct measure
exterior_bdries = MeshFunction('size_t', mesh,
                               mesh.topology().dim()-1, 0)
CompiledSubDomain("on_boundary").mark(exterior_bdries, 2)
FixedBC().mark(exterior_bdries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=exterior_bdries)

bcs = [DGDirichletBC(ds(1), gD_func), DGNeumannBC(ds(2), u)]

# List of fluxes to be used in the comparison
llf = LocalLaxFriedrichs(lambda u, n: u*n[0] + n[1])

# Automatic formulation of the DG spacetime Burgers problem
bo = SpacetimeBurgersOperator(mesh, V, bcs, flux=llf)
residual = bo.generate_fem_formulation(u, v)
J = derivative(residual, u, du)

for j in range(d_comp_n):
    solve(residual == 0, u)
    gD_func.u.vector()[:] = u.vector()[:]


x = np.linspace(0, 1, 100)
y = np.array(list(map(lambda x: u(x, d), x)))

plt.xlabel("$x$")
plt.ylabel("$u_h(x)$")
plt.plot(x, y)
plt.show()