from dolfin import *
from dolfin_dg import *
import math


# In this example we use dual weighted residual based error estimates to compute the drag
# coefficient of compressible flow around a NACA0012 airfoil.


parameters['std_out_all_processes'] = False
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 4


# Define a Nonlinear problem to assemble the residual and Jacobian
class Problem(NonlinearProblem):
    def __init__(self, a, L, bcs):
        self.a = a
        self.L = L
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        tic()
        assemble(self.a, tensor=A)
        elapsed_time = toc()
        info("Jacobian assembly time: %.3e s" % elapsed_time)


# Use a Newton solver with custom damping parameter
class CustomSolver(NewtonSolver):
    def __init__(self):
        self.solver = PETScKrylovSolver()
        NewtonSolver.__init__(self, mesh.mpi_comm(), self.solver, PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("pc_factor_mat_solver_package", "mumps")

        self.solver.set_from_options()

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem, iteration):
        tau = 1.0
        theta = min(sqrt(2.0*tau/norm(dx, norm_type="l2", mesh=V.mesh())), 1.0)
        info("Newton damping parameter: %.3e" % theta)
        x.axpy(-theta, dx)


mesh = Mesh("naca0012_coarse_mesh.xml")

# Polynomial order
poly_o = 1

# Initial inlet flow conditions
rho_0 = 1.0
M_0 = 0.5
Re_0 = 5e3
p_0 = 1.0
gamma = 1.4
attack = math.radians(2.0)

# Inlet conditions
c_0 = abs(gamma*p_0/rho_0)**0.5
Re = Re_0/(gamma**0.5*M_0)
n_in = Point(cos(attack), sin(attack))
u_ref = Constant(M_0*c_0)
rho_in = Constant(rho_0)
u_in = u_ref*as_vector((Constant(cos(attack)), Constant(sin(attack))))

# The initial guess used in the Newton solver. Here we use the inlet flow.
rhoE_in_guess = energy_density(p_0, rho_in, u_in)
gD_guess = as_vector((rho_in, rho_in*u_in[0], rho_in*u_in[1], rhoE_in_guess))

# Assign variable names to the inlet, outlet and adiabatic wall BCs. These indices
# will be used to define subsets of the boundary.
INLET = 1
OUTLET = 2
WALL = 3

# Write the adapted meshes
xdmf = XDMFFile("adapted_naca0012_meshes.xdmf")

# Record information in this lists
results = []

# Maximum number of refinement levels
n_ref_max = 7
for ref_level in range(n_ref_max):
    info("Refinement level %d" % ref_level)

    # Label the boundary compoonents of the mesh. Initially label all exterior facets
    # as the adiabatic wall, then label the exterior facets far from the airfoil
    # as the inlet and outlet based on the angle of attack.
    bdry_ff = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain("on_boundary").mark(bdry_ff, WALL)
    for f in facets(mesh):
        x = f.midpoint()
        if not f.exterior() or (x[0]*x[0] + x[1]*x[1]) < 4.0:
            continue
        bdry_ff[f] = INLET if f.normal().dot(n_in) < 0.0 else OUTLET

    ds = Measure('ds', domain=mesh, subdomain_data=bdry_ff)

    # Problem function space, (rho, rho*u1, rho*u2, rho*E)
    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
    info("Problem size: %d degrees of freedom" % V.dim())

    # Use the initial guess.
    u_vec = project(gD_guess, V); u_vec.rename("u", "u")
    v_vec = TestFunction(V)

    # The subsonic inlet, adiabatic wall and subsonic outlet conditions
    inflow = subsonic_inflow(rho_in, u_in, u_vec)
    no_slip_bc = no_slip(u_vec)
    outflow = subsonic_outflow(p_0, u_vec)

    # Assemble these conditions into DG BCs
    bcs = [DGDirichletBC(ds(INLET), inflow),
           DGAdiabticWallBC(ds(WALL), no_slip_bc),
           DGDirichletBC(ds(OUTLET), outflow)]

    # Construct the compressible Navier Stokes DG formulation, and compute the symbolic
    # Jacobian
    ce = CompressibleNavierStokesOperator(mesh, V, bcs, mu=1.0/Re)
    F = ce.generate_fem_formulation(u_vec, v_vec)
    J = derivative(F, u_vec)

    # Setup the problem and solve
    solver = CustomSolver()
    problem = Problem(J, F, [])
    solver.solve(problem, u_vec.vector())

    # Assemble variables required for the lift and drag computation
    n = FacetNormal(mesh)
    rho, u, E = flow_variables(u_vec)
    p = pressure(u_vec)
    l_ref = Constant(1.0)
    tau = 1.0/Re*(grad(u) + grad(u).T - 2.0/3.0*(tr(grad(u)))*Identity(2))
    C_infty = 0.5*rho_in*u_ref**2*l_ref

    # Assemble the homogeneity tensor with dot(grad(T), n) = 0 for use in the
    # adjoint consistent lift and drag formulation
    h = CellVolume(mesh)/FacetArea(mesh)
    sigma = Constant(20.0)*Constant(max(poly_o**2, 1))/h
    G_adiabitic = homogeneity_tensor(ce.F_v_adiabatic, u_vec)
    G_adiabitic = replace(G_adiabitic, {u_vec: no_slip_bc})

    # The drag coefficient
    psi_drag = as_vector((cos(attack), sin(attack)))
    drag = 1.0/C_infty*dot(psi_drag, p*n - tau*n)*ds(WALL)

    # The adjoint consistent drag coefficient
    z_drag = 1.0/C_infty*as_vector((0, psi_drag[0], psi_drag[1], 0))
    drag += inner(sigma*hyper_tensor_product(G_adiabitic, dg_outer(u_vec - no_slip_bc, n)), dg_outer(z_drag, n))*ds(WALL)

    # The lift coefficient
    psi_lift = as_vector((-sin(attack), cos(attack)))
    lift = 1.0/C_infty*dot(psi_lift, p*n - tau*n)*ds(WALL)

    result = (V.dim(), assemble(drag), assemble(lift))
    info("DoFs: %d, Drag: %.5e, Lift: %.5e" % result)
    results += [result]

    # If we're not on the last refinement level, apply dual-weighted-residual error
    # estimation refinement.
    if ref_level < n_ref_max - 1:
        est = NonlinearAPosterioriEstimator(J, F, drag, u_vec)
        markers = est.compute_cell_markers(FixedFractionMarker(frac=0.2))
        mesh = refine(mesh, cell_markers=markers)

    # Write the density and the adapted mesh
    xdmf.write(u_vec.sub(0), float(ref_level))

if MPI.rank(mesh.mpi_comm()) == 0:
    print("\n".join(map(lambda result: "DoFs: %d, Drag: %.5e, Lift: %.5e" % result, results)))