from dolfin import (
    parameters, BoxMesh, Point, TrialFunction, TestFunction,
    MeshFunction, Measure, derivative, solve, assemble, Constant, info,
    CompiledSubDomain, VectorFunctionSpace, CellVolume, FacetArea, inner,
    grad, dot, FacetNormal, tr, Identity, Function, variable, diff, ln, det,
    dS, dx, PETScOptions, XDMFFile, refine)

from dolfin_dg import homogeneity_tensor, DGFemSIPG
from dolfin_dg.dolfin import NonlinearAPosterioriEstimator, \
    FixedFractionMarkerParallel

parameters["std_out_all_processes"] = False
parameters["ghost_mode"] = "shared_facet"
parameters["refinement_algorithm"] = "plaza"

# Polynomial degree and quadrature order
poly_o = 1
parameters["form_compiler"]["quadrature_degree"] = 2*poly_o + 1

# Material Coefficients
E = 0.1e9
nu = 0.46

mesh = BoxMesh(Point(0., 0., 0.), Point(0.5, 0.5, 2.0), 4, 4, 12)

bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
top = CompiledSubDomain("near(x[2], side) && on_boundary", side=2.0)
clamp = Constant((0.0, 0.0, 0.0))
r = Constant((0.0, 0.07, -0.5))

ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
TOP, BOTTOM = 1, 2
top.mark(ff, TOP)
bottom.mark(ff, BOTTOM)
ds = Measure("ds", subdomain_data=ff)

# System constants
B = Constant((0.0, 0.0, 0.0))
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))


# Neo-Hookean strain energy function
def Psi(F):
    return (mu/2)*(tr(F.T*F) - 3) - mu*ln(det(F)) + (lmbda/2)*(ln(det(F)))**2


# First Piola-Kirchoff stress tensor
def F_v(u, grad_u):
    F = variable(Identity(3) + grad_u)
    return diff(Psi(F), F)


# Ouput file
xdmf = XDMFFile("displacement.xdmf")

dof_count = []
energies = []

# Goal oriented adaptive mesh refinement
n_ref_max = 2
for ref_level in range(n_ref_max):
    V = VectorFunctionSpace(mesh, "DG", poly_o)
    info(f"Refinement level: {ref_level}, Problem dim: {V.dim()}")

    # Mark the Dirichlet boundaries of the mesh
    ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    top.mark(ff, 1)
    bottom.mark(ff, 2)
    ds = Measure("ds", subdomain_data=ff)

    # We use fenicstools to perform mesh-to-mesh interpolation in parallel
    # between refinement levels
    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    u.rename("u", "u")

    # DGFEM formulation
    gamma = Constant(20.0)
    n = FacetNormal(mesh)
    h = CellVolume(mesh)/FacetArea(mesh)
    sig = gamma*max(poly_o**2, 1)/h

    G = homogeneity_tensor(F_v, u)
    vt = DGFemSIPG(F_v, u, v, sig, G, n)
    interior = vt.interior_residual(dS)
    exterior = vt.exterior_residual(clamp, ds(BOTTOM)) \
        + vt.exterior_residual(r, ds(TOP))

    F = inner(F_v(u, grad(u)), grad(v))*dx + interior + exterior - dot(B, v)*dx
    J = derivative(F, u, du)

    # Setup solver and solve the problem
    solve(F == 0, u, J=J)

    # Compute the internal strain energy functional
    W_int = Psi(Identity(3) + grad(u))
    W_int_total = assemble(W_int*dx)
    energies.append(W_int_total)
    dof_count.append(V.dim())

    # Write the displacement to file
    xdmf.write(u, float(ref_level))

    # If not on the last refinement level, employ adaptive refinement
    # based on a DWR posteriori error estimate
    if ref_level < n_ref_max - 1:
        info("Computing a posteriori error estimates")
        PETScOptions.set("dual_ksp_type", "preonly")
        PETScOptions.set("dual_pc_type", "lu")
        PETScOptions.set("dual_pc_factor_mat_solver_type", "mumps")

        est = NonlinearAPosterioriEstimator(J, F, W_int*dx, u,
                                            options_prefix="dual_")
        markers = est.compute_cell_markers(
            FixedFractionMarkerParallel(frac=0.1))
        mesh = refine(mesh, marker=markers, redistribute=True)

if mesh.mpi_comm().rank == 0:
    print("∫Ψ(u_h)dX:", energies)
    print("DoFs:", dof_count)

xdmf.close()
