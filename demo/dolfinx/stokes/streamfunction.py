import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfin_dg as dg

# C0-IPG formulation of the streamfunction formulation of the Stokes
# equations. Here we seek u such that
#
# -∇² u + ∇ p = f in Ω
#       u ⋅ n = 0 on ∂Ω
#
# Which may be reformulated in terms of the stream function u = ∇ × ψ
# so we seek ψ such that
#
#    ∇⁴ ψ = ∇ × f       in Ω
#       ψ = 0           on ∂Ω
# ∇ ψ ⋅ n = ∇ ψₛₒₗₙ ⋅ n on ∂Ω
#
# Where we approximate the manufactured solution ψₛₒₗₙ from an
# a priori known (also manufactured) velocity field uₛₒₗₙ
#
#  - ∇² ψₛₒₗₙ = ∇ × uₛₒₗₙ in Ω
#       ψₛₒₗₙ = 0         on ∂Ω
#
# We weakly enforce the C1 continuity by applying a DG method to the
# second integration by parts of the biharmonic operator.

ele_ns = [8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
errorpsil2 = np.zeros(len(ele_ns))
errorpsih1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))

p = 2

for j, n_ele in enumerate(ele_ns):
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,
        [np.array((-1.0, -1.0)), np.array((1.0, 1.0))],
        [n_ele, n_ele])
    PHI = dolfinx.fem.FunctionSpace(mesh, ("CG", p))
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", p-1))

    n = ufl.FacetNormal(mesh)
    psi = dolfinx.fem.Function(PHI)
    phi = ufl.TestFunction(PHI)

    # Velocity solution where u ⋅ n = 0 on the boundary
    x = ufl.SpatialCoordinate(mesh)
    u_soln = ufl.as_vector((
            2*x[1]*(1.0 - x[0]*x[0]),
            -2*x[0]*(1.0 - x[1]*x[1])))

    # Compute the approximation of the true solution of ψ
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim-1,
        marker=lambda x: np.ones_like(x[0], dtype=np.int8))
    dofs = dolfinx.fem.locate_dofs_topological(
        PHI, mesh.topology.dim-1, facets)
    zero_bc = dolfinx.fem.dirichletbc(0.0, dofs, PHI)

    psi_soln = dolfinx.fem.Function(PHI)
    F_psi = ufl.dot(ufl.grad(psi_soln), ufl.grad(phi)) * ufl.dx - ufl.curl(u_soln) * phi * ufl.dx

    problem = dolfinx.fem.petsc.NonlinearProblem(F_psi, psi_soln, [zero_bc])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    # We can customize the linear solver used inside the NewtonSolver by
    # modifying the PETSc options
    opts = PETSc.Options()
    option_prefix = solver.krylov_solver.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    solver.krylov_solver.setFromOptions()

    solver.solve(psi_soln)

    # The second order viscous flux
    def F_v(u, div_grad_u):
        return div_grad_u

    # Fourth order DG discretisation
    G = dg.homogeneity_tensor(
        F_v, psi, differential_operator=lambda u: ufl.div(ufl.grad(u)))
    sigma = dg.generate_default_sipg_penalty_term(
        psi, C_IP=dolfinx.fem.Constant(mesh, 1e1))
    fo = dg.DGClassicalFourthOrderDiscretisation(F_v, psi, phi, sigma, G, n, -1)

    # Finite element residual
    f = ufl.curl(-ufl.div(ufl.grad(u_soln)))
    F = ufl.inner(ufl.div(ufl.grad(psi)), ufl.div(ufl.grad(phi))) * ufl.dx - f * phi * ufl.dx
    F += fo.interior_residual(ufl.dS)

    # Weakly enforce ∇ ψ ⋅ n = ∇ ψₛₒₗₙ ⋅ n on ∂Ω
    F += fo.exterior_residual(psi_soln, ufl.ds)

    # Solve the linear system
    J = ufl.derivative(F, psi)

    problem = dolfinx.fem.petsc.NonlinearProblem(F, psi, [zero_bc])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    opts = PETSc.Options()
    option_prefix = solver.krylov_solver.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    solver.krylov_solver.setFromOptions()

    solver.solve(psi)

    # Interpolate the streamfunction in the (p-1) velocity space
    u_expr = dolfinx.fem.Expression(ufl.curl(psi), V.element.interpolation_points())
    u = dolfinx.fem.Function(V)
    u.interpolate(u_expr)

    # Compute error
    errorl2[j] = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(
            (u - u_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)
    errorh1[j] = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(
            ufl.grad(u - u_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)
    errorpsil2[j] = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(
            (psi - psi_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)
    errorpsih1[j] = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(
            ufl.grad(psi - psi_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)

    h_measure = dolfinx.cpp.mesh.h(
        mesh, 2, np.arange(mesh.topology.connectivity(2, 0).num_nodes,
                           dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hsizes[j] = hmin


if mesh.comm.rank == 0:
    print("L2 u convergence rates: "
          + str(np.log(errorl2[:-1] / errorl2[1:])
                / np.log(hsizes[:-1] / hsizes[1:])))
    print("H1 u convergence rates: "
          + str(np.log(errorh1[:-1] / errorh1[1:])
                / np.log(hsizes[:-1] / hsizes[1:])))
    print("L2 psi convergence rates: "
          + str(np.log(errorpsil2[:-1] / errorpsil2[1:])
                / np.log(hsizes[:-1] / hsizes[1:])))
    print("H1 psi convergence rates: "
          + str(np.log(errorpsih1[:-1] / errorpsih1[1:])
                / np.log(hsizes[:-1] / hsizes[1:])))
