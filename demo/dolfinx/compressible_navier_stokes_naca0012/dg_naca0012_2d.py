import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import dolfin_dg
import dolfin_dg.dolfinx.dwr
import dolfin_dg.dolfinx.mark

import generate_mesh

# In this example we use dual weighted residual based error estimates
# to compute the drag coefficient of compressible flow around a NACA0012
# aerofoil.

def info(*msg):
    PETSc.Sys.Print(", ".join(map(str, msg)))

mesh = generate_mesh.generate_naca_4digit(
    MPI.COMM_WORLD, *generate_mesh.parse_naca_digits("0012"), rounded=False,
    gmsh_options={"Mesh.MeshSizeFromCurvature": 80})

# Polynomial order
poly_o = 1

# Initial inlet flow conditions
rho_0 = 1.0
M_0 = 0.5
Re_0 = 5e3
p_0 = 1.0
gamma = 1.4
attack = np.radians(2.0)

# Inlet conditions
c_0 = abs(gamma*p_0/rho_0)**0.5
Re = Re_0/(gamma**0.5*M_0)
n_in = dolfinx.fem.Constant(
    mesh, np.array([np.cos(attack), np.sin(attack)], dtype=np.double))
u_ref = dolfinx.fem.Constant(mesh, M_0*c_0)
rho_in = dolfinx.fem.Constant(mesh, rho_0)
u_in = u_ref * n_in

# The initial guess used in the Newton solver. Here we use the inlet flow.
rhoE_in_guess = dolfin_dg.aero.energy_density(p_0, rho_in, u_in, gamma)
gD_guess = ufl.as_vector((rho_in, rho_in*u_in[0], rho_in*u_in[1], rhoE_in_guess))

# Assign variable names to the inlet, outlet and adiabatic wall BCs. These
# indices will be used to define subsets of the boundary.
INLET = 3
OUTLET = 4
WALL = 2

# Record information in this list
results = []

# Maximum number of refinement levels
n_ref_max = 3
for ref_level in range(n_ref_max):
    info(f"Refinement level {ref_level}")

    # Label the boundary components of the mesh. Initially label all exterior
    # facets as the adiabatic wall, then label the exterior facets far from
    # the aerofoil as the inlet and outlet based on the angle of attack.
    bdry_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim-1,
        lambda x: np.full_like(x[0], 1, dtype=np.int8))

    midpoints = dolfinx.mesh.compute_midpoints(
        mesh, mesh.topology.dim-1, bdry_facets)
    f_normals = dolfinx.cpp.mesh.cell_normals(
        mesh._cpp_object, mesh.topology.dim-1, bdry_facets)

    inner_facets = np.linalg.norm(midpoints, axis=1) < 10.0
    inlet = np.dot(f_normals[:,:2], n_in.value) < 0.0

    values = np.zeros(midpoints.shape[0], dtype=np.int32)
    values[inlet] = INLET
    values[~inlet] = OUTLET
    values[inner_facets] = WALL

    fts = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, bdry_facets, values)

    ds = ufl.Measure('ds', domain=mesh, subdomain_data=fts)

    # Problem function space, (rho, rho*u1, rho*u2, rho*E)
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", poly_o), dim=4)
    n_dofs = mesh.comm.allreduce(
        V.dofmap.index_map.size_local * V.dofmap.index_map_bs, MPI.SUM)
    info(f"Problem size: {n_dofs} degrees of freedom")

    u_vec = dolfinx.fem.Function(V, name="u")
    if ref_level == 0:
        # Use the initial guess.
        u_vec.interpolate(
            dolfinx.fem.Expression(gD_guess, V.element.interpolation_points()))
    else:
        # Initial guess by interpolating from old mesh to new mesh
        interp_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            u_vec.function_space.mesh._cpp_object,
            u_vec.function_space.element,
            u_vec_old.function_space.mesh._cpp_object, padding=1e-4)
        u_vec.interpolate(u_vec_old, nmm_interpolation_data=interp_data)
    u_vec.x.scatter_forward()
    v_vec = ufl.TestFunction(V)

    # The subsonic inlet, adiabatic wall and subsonic outlet conditions
    inflow = dolfin_dg.aero.subsonic_inflow(rho_in, u_in, u_vec, gamma)
    no_slip_bc = dolfin_dg.aero.no_slip(u_vec)
    outflow = dolfin_dg.aero.subsonic_outflow(p_0, u_vec, gamma)

    # Assemble these conditions into DG BCs
    bcs = [dolfin_dg.DGDirichletBC(ds(INLET), inflow),
           dolfin_dg.DGAdiabticWallBC(ds(WALL), no_slip_bc),
           dolfin_dg.DGDirichletBC(ds(OUTLET), outflow)]

    # Construct the compressible Navier Stokes DG formulation, and compute
    # the symbolic Jacobian
    h = ufl.CellVolume(mesh)/ufl.FacetArea(mesh)
    ce = dolfin_dg.CompressibleNavierStokesOperator(mesh, V, bcs, mu=1.0/Re)
    F = ce.generate_fem_formulation(u_vec, v_vec, h_measure=h, c_ip=20.0)
    J = ufl.derivative(F, u_vec)

    # Set up the problem and solve
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u_vec, J=J)
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)

    def updater(solver, dx, x):
        # TODO: This causes a memory leak
        tau = 1.0
        theta = min((2.0*tau/dx.norm())**0.5, 1.0)
        x.axpy(-theta, dx)

    solver.set_update(updater)

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()
    solver.solve(u_vec)
    u_vec.x.scatter_forward()

    # Assemble variables required for the lift and drag computation
    n = ufl.FacetNormal(mesh)
    rho, u, E = dolfin_dg.aero.flow_variables(u_vec)
    p = dolfin_dg.aero.pressure(u_vec, gamma)
    l_ref = dolfinx.fem.Constant(mesh, 1.0)
    tau = 1.0/Re*(ufl.grad(u) + ufl.grad(u).T - 2.0/3.0*(ufl.tr(ufl.grad(u)))*ufl.Identity(mesh.geometry.dim))
    C_infty = 0.5*rho_in*u_ref**2*l_ref

    # Assemble the homogeneity tensor with dot(grad(T), n) = 0 for use in the
    # adjoint consistent lift and drag formulation
    sigma = dolfinx.fem.Constant(mesh, 20.0)*dolfinx.fem.Constant(mesh, float(max(poly_o**2, 1)))/h
    import dolfin_dg.primal.aero
    F_v_adiabatic = dolfin_dg.primal.aero.compressible_navier_stokes_adiabatic_wall(
        u_vec, v_vec, gamma=gamma, mu=1/Re).F_vec[1]
    G_adiabitic = dolfin_dg.math.homogeneity_tensor(F_v_adiabatic, u_vec)
    G_adiabitic = ufl.replace(G_adiabitic, {u_vec: no_slip_bc})

    # The drag coefficient
    psi_drag = n_in
    drag = 1.0/C_infty*ufl.dot(psi_drag, p*n - tau*n)*ds(WALL)

    # The adjoint consistent drag coefficient
    z_drag = 1.0/C_infty*ufl.as_vector((0, psi_drag[0], psi_drag[1], 0))
    drag += ufl.inner(
        sigma*dolfin_dg.math.hyper_tensor_product(G_adiabitic, ufl.outer(u_vec - no_slip_bc, n)),
        ufl.outer(z_drag, n))*ds(WALL)

    # The lift coefficient
    psi_lift = dolfinx.fem.Constant(
        mesh, np.array([-np.sin(attack), np.cos(attack)], dtype=np.double))
    lift = 1.0/C_infty*ufl.dot(psi_lift, p*n - tau*n)*ds(WALL)

    drag_val = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(drag)), MPI.SUM)
    lift_val = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(lift)), MPI.SUM)

    info(f"DoFs: {n_dofs}, Drag: {drag_val:.5e}, Lift: {lift_val:.5e}")
    results += [(n_dofs, drag_val, lift_val)]

    with dolfinx.io.VTXWriter(
            mesh.comm, f"adapted_naca0012_meshes_{ref_level}.bp",
            [u_vec.sub(0).collapse()], "bp4") as f:
        f.write(0.0)

    # If we're not on the last refinement level, apply goal oriented
    # dual-weighted-residual error estimation refinement.
    if ref_level < n_ref_max - 1:
        V_star = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', poly_o+1), dim=4)

        n_dofs = mesh.comm.allreduce(
            V_star.dofmap.index_map.size_local * V_star.dofmap.index_map_bs, MPI.SUM)
        info(f"Computing a posteriori error estimates:\n"
             f"Dual space problem size: {n_dofs} degrees of freedom")

        est = dolfin_dg.dolfinx.dwr.NonlinearAPosterioriEstimator(
            J, F, drag, u_vec, V_star)
        indicators = est.compute_indicators(petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"})
        cell_markers = dolfin_dg.dolfinx.mark.maximal_indices_fraction(
            indicators, 0.1)

        info("Refining mesh")
        edges_to_ref = dolfinx.mesh.compute_incident_entities(
            mesh.topology, cell_markers, mesh.topology.dim, 1)
        new_mesh, _, _ = dolfinx.cpp.refinement.refine_plaza(
            mesh._cpp_object, edges_to_ref, True,
            dolfinx.mesh.RefinementOption.none)
        new_mesh = dolfinx.mesh.Mesh(
            new_mesh, ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element()))

        mesh = new_mesh
        u_vec_old = u_vec

if mesh.comm.rank == 0:
    for result in results:
        print("DoFs: {:d}, Drag: {:.5e}, Lift: {:.5e}".format(*result))
