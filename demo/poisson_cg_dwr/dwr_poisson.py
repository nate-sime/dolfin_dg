from dolfin import *
from dolfin_dg import *
import matplotlib.pyplot as plt

if MPI.size(MPI.comm_world) > 1:
    NotImplementedError("Dual weighted residual (DWR) functionality not fully supported in parallel.")

# Demo taken from FEniCS course Lecture 11. A. Logg and M. Rognes
# https://fenicsproject.org/pub/course/lectures/2013-11-logg-fcc/lecture_11_error_control.pdf
mesh = UnitSquareMesh(2, 2)

poly_o = 2

# True error
j = 3.56521530256e-05

# Forcing term
f = Expression("exp(-100.0*(x[0]*x[0] + x[1]*x[1]))", degree=poly_o+1)

# First, use adaptive DWR-based refinement and record the error vs DoF count
dwr_errors = []
dwr_dofs = []

for it in range(7):
    V = FunctionSpace(mesh, 'CG', poly_o)
    u, v = TrialFunction(V), TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    soln = Function(V)
    bcs = DirichletBC(V, Constant(0.0), 'on_boundary')
    solve(a == L, soln, bcs)

    dwr_errors.append(abs(assemble(soln*dx) - j))

    dwr_dofs.append(V.dim())

    j_h = Constant(10.0)*u*dx

    lape = LinearAPosterioriEstimator(a, L, j_h, soln, bcs)
    markers = lape.compute_cell_markers(FixedFractionMarker(frac=0.2))

    mesh = refine(mesh, markers, redistribute=True)

plt.loglog(dwr_dofs, dwr_errors)


# Second perform h-refinement and record error vs DoF count
href_errors = []
href_dofs = []

for it in range(1, 6):
    mesh = UnitSquareMesh(2**it, 2**it)
    V = FunctionSpace(mesh, 'CG', poly_o)
    u, v = TrialFunction(V), TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    soln = Function(V)
    bcs = DirichletBC(V, Constant(0.0), 'on_boundary')
    solve(a == L, soln, bcs)

    href_errors.append(abs(assemble(soln*dx) - j))
    href_dofs.append(V.dim())

plt.loglog(href_dofs, href_errors)
plt.xlabel("DoFs")
plt.ylabel("$|J(u) - J(u_h)|$")
plt.legend(["DWR $h-$refinement", "$h-$refinement"])

plt.show()