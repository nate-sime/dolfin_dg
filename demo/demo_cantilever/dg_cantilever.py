import numpy as np
from dolfin import *
from dolfin_dg import *

# This demo reproduces the numerical experiment shown in Section 4.1 of
# L. Noels and R. Radovitzky, A general discontinuous galerkin method for finite hyperelasticity. formulation
# and numerical applications, Internat. J. Numer. Methods Engrg. 68 (2006), no. 1, 64–97.
# Here we compute the small-strain deflection of a cantilever.

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["std_out_all_processes"] = False
parameters["ghost_mode"] = "shared_facet"

# Quadrature in 3D is expensive, try to use optimal order
poly_o = 2
parameters["form_compiler"]["quadrature_degree"] = 2*poly_o + 1

# Material coefficients
Ey = 200e9
nu = 0.3

mu = Constant(Ey/(2*(1 + nu)))
lmbda = Constant(Ey*nu/((1 + nu)*(1 - 2*nu)))

# Geometry dimensions
L = 1.0
beta = 0.1

# Functionals stored for mesh convergence study
h_max = []
tip_disp = []
W_int = []


for n in range(1, 7):
    # Create mesh and define function space
    mesh = BoxMesh(Point(0, 0, 0), Point(beta, beta, L), n, n, 3*n)

    V = VectorFunctionSpace(mesh, "DG", poly_o)
    info("Initialising system with %d DoF" % V.dim())

    # The left and right face subdomains used in the boundary defintions
    left = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
    right = CompiledSubDomain("near(x[2], side) && on_boundary", side=L)
    c = Constant((0.0, 0.0, 0.0))

    # A FacetFunction
    ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    LEFT, RIGHT = 1, 2
    left.mark(ff, LEFT)
    right.mark(ff, RIGHT)
    ds = Measure("ds", subdomain_data=ff)

    # Define functions
    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    B = Constant((0.0, 0.0, 0.0))

    # Distributed force
    force = 10e3
    T = Constant((0.0, force/beta**2, 0.0))

    # Neo-Hookean strain energy function
    def Psi(F):
        return (mu/2)*(tr(F.T*F) - 3) - mu*ln(det(F)) + (lmbda/2)*(ln(det(F)))**2

    # First Piola-Kirchoff stress tensor
    def F_v(u, grad_u):
        F = variable(Identity(3) + grad_u)
        return diff(Psi(F), F)

    n = FacetNormal(mesh)
    he = CellVolume(mesh)/FacetArea(mesh)
    gamma = Constant(200.0)
    sig = gamma*max(poly_o**2, 1)/he

    G = homogeneity_tensor(F_v, u)

    vt = DGFemSIPG(F_v, u, v, sig, G, n)
    interior = vt.interior_residual(dS)
    exterior = vt.exterior_residual(c, ds(LEFT)) + vt.neumann_residual(T, ds(RIGHT))

    F = inner(F_v(u, grad(u)), grad(v))*dx - dot(B, v)*dx + interior + exterior

    # Compute Jacobian of F
    J = derivative(F, u, du)

    solve(F == 0, u, bcs=[], J=J, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

    # Store mesh size and strain energy
    h_max.append(mesh.hmax())
    W_int.append(assemble(Psi(Identity(3) + grad(u))*dx))

    # Find which process owns the tip of the cantilever and compute the displacement there.
    tip_cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(beta/2, beta/2, L))
    local_tip_disp = u(beta/2, beta/2, L)[1] if distance < DOLFIN_EPS else None

    # Once found, broadcast the displacement at the tip to process 0.
    comm = mesh.mpi_comm().tompi4py()
    computed_tip_displacements = comm.gather(local_tip_disp)
    if comm.rank == 0:
        # If process 0 has received more than one valid tip displacement from the other processes,
        # this means that the tip lies on both a process boundary and on a point adjacent to two (or more)
        # cells. We must assert that all the computed tip displacements from these processes
        # are approximately equivalent.
        valid_displacements = np.array([y for y in computed_tip_displacements if y is not None], dtype=np.double)
        assert np.all(np.abs(valid_displacements[0] - valid_displacements) < 1e-9)
        tip_disp.append(valid_displacements[0])


if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    # Compute internal strain energy and tip displacement error
    error_wint = np.abs(np.array(W_int, dtype=np.double) - 10.0)
    error_tip_disp = np.abs(np.array(tip_disp, dtype=np.double) - 2e-3)
    h_max = np.array(h_max, dtype=np.double)

    print("Mesh sizes:", h_max)
    print("Internal strain energies:", W_int)
    print("Tip displacements:", tip_disp)

    print("\u222B\u03A8(u_h)dX convergence rate:", np.log(error_wint[0:-1]/error_wint[1:])/np.log(h_max[0:-1]/h_max[1:]))
    print("u_h(\u03B2/2,\u03B2/2,L) convergence rate:", np.log(error_tip_disp[0:-1]/error_tip_disp[1:])/np.log(h_max[0:-1]/h_max[1:]))