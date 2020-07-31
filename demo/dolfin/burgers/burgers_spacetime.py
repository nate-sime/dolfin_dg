from dolfin import *

import ufl

__author__ = 'njcs4'

# Note that DG0 is essentially the finite volume method. DGp (p > 0) is a
# true high order DG method, but will be unstable at the shock. This is where
# hp-adaptivity would be a benefit. Moving to high order DG, be careful how
# long your domain is in time to ensure a stable solution, as the shock
# becomes more prominent

parameters["ghost_mode"] = "shared_facet"

# a = max distance
# d = max time
a, d = 1.0, 0.5

class FixedBC(SubDomain):
    def inside(self, x, on):
        return (abs(x[0]) < DOLFIN_EPS or abs(x[1]) < DOLFIN_EPS) and on

class FreeBC(SubDomain):
    def inside(self, x, on):
        return (abs(x[0] - a) < DOLFIN_EPS or abs(x[1] - d) < DOLFIN_EPS) and on

# Mesh and function space.
mesh = RectangleMesh(Point(0.0, 0.0), Point(a, d), 64, 64)
V = FunctionSpace(mesh, 'DG', 1)
du = TrialFunction(V)

# Set up Dirichlet BC
gD = Expression('max(sin(2.0*pi*x[0]/a), 0.0)',
                a=a, element=V.ufl_element())
u = project(gD, V)
v = TestFunction(V)
n = FacetNormal(mesh)

# Establish exterior boundary components and construct measure
exterior_bdries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
fixed_ds = FixedBC()
fixed_ds.mark(exterior_bdries, 1)
free_ds = FreeBC()
free_ds.mark(exterior_bdries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=exterior_bdries)

# Define convective flux tensor
def F_c(u):
    return as_vector((u**2/2, u))

# Define local-Lax Friedrichs flux
# alpha is the maximum of the flux Jacobian eigenvalues
def H(u_p, u_m, n):
    # note that n^+ = -n^-
    alpha = ufl.Max(abs(u_p*n[0] + n[1]), abs(-u_m*n[0] - n[1]))
    return 0.5*(dot(F_c(u_p), n) + dot(F_c(u_m), n) + alpha*(u_p - u_m))

# Volume integration terms
convective_domain = -inner(F_c(u), grad(v))*dx

# Interior boundary face integrals
convective_interior = H(u('+'), u('-'), n('+'))*(v('+') - v('-'))*dS

# Exterior Dirichlet condition
convective_exterior = H(u, gD, n)*v*ds(1)

# Exterior Neumann condition
convective_exterior += H(u, u, n)*v*ds(2)

residual = convective_domain + convective_interior + convective_exterior
J = derivative(residual, u, du)

class BurgersProblem(NonlinearProblem):
    def F(self, b, x):
        assemble(residual, tensor=b)
    def J(self, A, x):
        assemble(J, tensor=A)

burgers = BurgersProblem()
solver = PETScSNESSolver('newtonls')

solver.solve(BurgersProblem(), u.vector())

# Project to piecewise linears so we get a surface plot
plot(u, backend='matplotlib')
import matplotlib.pyplot as plt
plt.show()