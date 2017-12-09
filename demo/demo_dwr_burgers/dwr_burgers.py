from dolfin import *
from dolfin_dg import *
import matplotlib.pyplot as plt

# This demo aims to reproduce numerical example 7.1 Hartmann and Houston 2002

parameters["ghost_mode"] = "shared_facet"
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

if MPI.size(mpi_comm_world()) > 1:
    error("Dual weighted residual (DWR) functionality not fully supported in parallel.")

mesh = RectangleMesh(Point(0., 0.), Point(3., 2.), 8, 6)

# Dirichlet boundary condition
gD = Expression("2.0/(1.0 + pow(x[0], 3))*pow(sin(pi*x[0]), 2)", degree=6)

# Storage of dof and true error J(u) - J(u_h)
dofs = []
errors = []

# Number of adaptive refinement levels
N = 8

for it in range(N):
    V = FunctionSpace(mesh, 'DG', 1)
    if it == 0:
        u = Function(V)
    else:
        u = interpolate(u, V)

    v = TestFunction(V)

    ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain("on_boundary").mark(ff, 1)
    CompiledSubDomain("near(x[1], 0.0) || near(x[0], 0.0)").mark(ff, 2)
    ds = Measure("ds", subdomain_data=ff)

    bcs = [DGDirichletBC(ds(2), gD), DGNeumannBC(ds(1), Constant(0.0))]

    # Automatically construct FEM formulation
    so = SpacetimeBurgersOperator(mesh, V, bcs)
    F = so.generate_fem_formulation(u, v)

    du = TrialFunction(V)
    J = derivative(F, u, du)
    solve(F == 0, u, J=J, solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})

    # Compute true error
    errors.append(abs(0.451408206331223 - u(1.95, 1.35)))
    dofs.append(V.dim())

    # Point of interest formulated using a mollified delta-function
    x = SpatialCoordinate(mesh)
    x0 = Constant((1.95, 1.35))

    pos = x - x0
    eps = Constant(1e-3)
    delta_x0 = 1.0/pi*eps/((pos[0]**2 + pos[1]**2)**0.5 + eps**2)

    # j(u_h) = \delta(x - x_0) u_h * dx
    j = delta_x0*u*dx

    if it == (N - 1):
        break

    # Compute dual-weighted residual error and refine the mesh
    est = NonlinearAPosterioriEstimator(J, F, j, u)
    markers = est.compute_cell_markers(FixedFractionMarker(frac=0.1))
    mesh = refine(mesh, cell_markers=markers, redistribute=True)

plt.figure()
plt.loglog(dofs, errors)
plt.ylabel(r"$|J(u) - J(u_h)|$")
plt.xlabel("DoFs")

plt.figure()
plot(mesh)

plt.figure()
plot(u)
plt.show()