import dolfinx.mesh
import dolfinx.fem
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfin_dg.primal
import dolfin_dg.primal.stokes
import dolfin_dg.dolfinx


# Printing alias which flushes in parallel processing
def info(msg):
    PETSc.Sys.Print(msg)


comm = MPI.COMM_WORLD
p_order = 2
dirichlet_id, neumann_id = 1, 2


# a priori known velocity and pressure solutions
def u_analytical(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[0] = -(x[1] * np.cos(x[1]) + np.sin(x[1])) * np.exp(x[0])
    vals[1] = x[1] * np.sin(x[1]) * np.exp(x[0])
    return vals


def p_analytical(x):
    return 2.0 * np.exp(x[0]) * np.sin(x[1]) + 1.5797803888225995912 / 3.0


# Run convergence rate experiment on all matrix types
l2errors_u = []
l2errors_p = []
hs = []

for run_no, n_ele in enumerate([8, 16, 32]):
    mesh = dolfinx.mesh.create_unit_square(
        comm, n_ele, n_ele,
        cell_type=dolfinx.cpp.mesh.CellType.triangle,
        ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

    h_measure = dolfinx.cpp.mesh.h(
        mesh._cpp_object, 2, np.arange(
            mesh.topology.index_map(2).size_local, dtype=np.int32))
    hmin = comm.allreduce(h_measure.min(), op=MPI.MIN)

    # Higher order FE spaces for interpolation of the true solution
    V_high = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", p_order + 2))
    Q_high = dolfinx.fem.FunctionSpace(mesh, ("DG", p_order + 1))

    u_soln = dolfinx.fem.Function(V_high)
    u_soln.interpolate(u_analytical)

    p_soln = dolfinx.fem.Function(Q_high)
    p_soln.interpolate(p_analytical)

    # Problem FE spaces and FE functions
    Ve = ufl.VectorElement("DG", mesh.ufl_cell(), p_order)
    Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), p_order-1)

    W = dolfinx.fem.FunctionSpace(mesh, ufl.MixedElement([Ve, Qe]))
    U = dolfinx.fem.Function(W, name="U")
    u, p = ufl.split(U)
    dU = ufl.TrialFunction(W)
    VV = ufl.TestFunction(W)
    v, q = ufl.split(VV)

    # Label Dirichlet and Neumann boundary components
    dirichlet_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                np.isclose(x[1], 0.0)))
    neumann_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.logical_or(np.isclose(x[0], 1.0),
                                np.isclose(x[1], 1.0)))

    bc_facet_ids = np.concatenate((dirichlet_facets, neumann_facets))
    bc_idxs = np.concatenate((np.full_like(dirichlet_facets, dirichlet_id),
                              np.full_like(neumann_facets, neumann_id)))
    sorted_idx = np.argsort(bc_facet_ids)
    facets = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1,
        bc_facet_ids[sorted_idx], bc_idxs[sorted_idx])
    ds = ufl.Measure("ds", subdomain_data=facets)
    dsD, dsN = ds(dirichlet_id), ds(neumann_id)

    # Stokes system nonlinear viscosity model
    def eta(u):
        return 1 + 1e-1 * ufl.dot(u, u)

    def stress(u, p):
        return 2*eta(u)*ufl.sym(ufl.grad(u)) - p*ufl.Identity(mesh.geometry.dim)

    # Momentum equation
    fos_mom = dolfin_dg.primal.stokes.stokes_stress(u, v, p, eta(u))
    facet_n = ufl.FacetNormal(mesh)
    gN = stress(u_soln, p_soln)*facet_n

    f = -ufl.div(stress(u_soln, p_soln))
    F = - ufl.inner(f, v) * ufl.dx
    # F += fos_mom.domain()
    # F += - ufl.inner(gN, v) * dsN
    #
    # alpha = dolfin_dg.generate_default_sipg_penalty_term(u)
    # F += fos_mom.interior([alpha("+")])
    # F += fos_mom.exterior([alpha], u_soln, ds=dsD)
    #
    # # Mass equation (integrate by parts twice: no penalty parameter)
    # fos_mass = dolfin_dg.primal.stokes.stokes_continuity(u, q)
    # F += fos_mass.domain()
    # F += fos_mass.interior([])
    # F += fos_mass.exterior([], u_soln, ds=dsD)

    bcs = [dolfin_dg.DGDirichletBC(dsD, u_soln),
           dolfin_dg.DGNeumannBC(dsN, gN)]
    stokes_op = dolfin_dg.operators.StokesOperator(None, None, bcs, None)
    F += stokes_op.generate_fem_formulation(u, v, p, q, eta)
    # quit()

    J = ufl.derivative(F, U)

    # Setup SNES solver
    snes = PETSc.SNES().create(comm)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()

    # Setup nonlinear problem
    F, J = map(dolfinx.fem.form, (F, J))
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(
        F, J, U, [])

    # Construct linear system data structures
    snes.setFunction(
        problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(
        problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J), P=None)
    soln_vector = U.vector

    # Set solver options
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    # Solve and check convergence
    snes.solve(None, soln_vector)
    snes_converged = snes.getConvergedReason()
    ksp_converged = snes.getKSP().getConvergedReason()
    if snes_converged < 1 or ksp_converged < 1:
        info(f"SNES converged reason: {snes_converged}")
        info(f"KSP converged reason: {ksp_converged}")

    # Computer error
    l2error_u = comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    l2error_p = comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((p - p_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)

    hs.append(hmin)
    l2errors_u.append(l2error_u)
    l2errors_p.append(l2error_p)

# Compute convergence rates
l2errors_u = np.array(l2errors_u)
l2errors_p = np.array(l2errors_p)
hs = np.array(hs)

hrates = np.log(hs[:-1] / hs[1:])
rates_u = np.log(l2errors_u[:-1] / l2errors_u[1:]) / hrates
rates_p = np.log(l2errors_p[:-1] / l2errors_p[1:]) / hrates
info(f"l2 u: {l2errors_u}")
info(f"rates u: {rates_u}")
info(f"l2 p: {l2errors_p}")
info(f"rates p: {rates_p}")
