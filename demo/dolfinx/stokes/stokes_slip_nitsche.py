import dolfinx.fem
import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
from petsc4py import PETSc

import dolfin_dg
import dolfin_dg.dolfinx

"""
This demo is the dolfinx implementation of Example 1
presented in Sime & Wilson (2020) https://arxiv.org/abs/2001.10639
which was originally written for dolfin-x.

To see close to optimal rates of convergence in case 2, we recommend
using higher order quadrature and an element degree approximation of
the true solution > p + 4. These parameters lead to long compilation
times and are switched off by default so that CI tests are not so
expensive.
"""

ele_ns = [8, 16, 32]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
errorpl2 = np.zeros(len(ele_ns))
errorph1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))

case = 1
if case == 1:
    # Exact representation in Q2 & P3
    u_soln_f = lambda x: np.stack((2*x[1]*(1.0 - x[0]*x[0]),
                                   -2*x[0]*(1.0 - x[1]*x[1])))
elif case == 2:
    # Care must be taken in the event of a singularity at r = 0
    u_soln_f = lambda x: np.stack((-x[1]*(x[0]*x[0] + x[1]*x[1])**0.5,
                                   x[0]*(x[0]*x[0] + x[1]*x[1])**0.5))

for j, ele_n in enumerate(ele_ns):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [(-1, -1), (1, 1)], [ele_n, ele_n],
        cell_type=dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

    We = ufl.MixedElement([ufl.VectorElement("CG", mesh.ufl_cell(), 2),
                           ufl.FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = dolfinx.fem.FunctionSpace(mesh, We)
    V, V2W = W.sub(0).collapse()
    Q, Q2W = W.sub(1).collapse()
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    # Define the velocity solution on an appropriately precise element
    u_soln = ufl.as_vector(u_soln_f(x))
    p_soln = dolfinx.fem.Constant(mesh, 0.0)

    U = dolfinx.fem.Function(W)
    u, p = ufl.split(U)

    V = ufl.TestFunction(W)
    v, q = ufl.split(V)

    # Prevent initial singularity in viscosity
    U.sub(0).interpolate(lambda x: np.stack((x[0], x[1])))

    # Nonlinear viscosity model
    def eta(u):
        return 1 + ufl.sqrt(1 + ufl.inner(ufl.grad(u), ufl.grad(u)))**-1

    # Problem definition in terms of a viscous flux operator F_v(u, grad(u))
    def F_v(u, grad_u, p_local=None):
        if p_local is None:
            p_local = p
        return 2 * eta(u) * ufl.sym(grad_u) - p_local * ufl.Identity(2)

    f = -ufl.div(F_v(u_soln, ufl.grad(u_soln), p_soln))
    g_tau = F_v(u_soln, ufl.grad(u_soln), p_soln) * n

    F = (
                ufl.inner(F_v(u, ufl.grad(u)), ufl.grad(v)) - ufl.dot(f, v)
                + ufl.div(u) * q
        ) * ufl.dx

    # We can choose a direction in which to enforce a Nitsche boundary
    # condition
    tau = ufl.as_vector((n[1], -n[0]))
    stokes_nitsche = dolfin_dg.StokesNitscheBoundary(F_v, u, p, v, q, delta=-1)
    F += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ufl.ds, tau=tau)

    zero = dolfinx.fem.Function(Q)
    zero.x.set(0.0)
    bc_dofs = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x.T, [1, 1, 0]).all(axis=1))
    bcs = [dolfinx.fem.dirichletbc(zero, bc_dofs, W.sub(1))]

    J = ufl.derivative(F, U)
    F, J = map(dolfinx.fem.form, (F, J))
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(
        F, J, U, bcs)

    # Setup SNES solver
    snes = PETSc.SNES().create(mesh.comm)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.setFunction(
        problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(
        problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J), P=None)
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    # Solve and check convergence
    snes.solve(None, U.vector)
    snes_converged = snes.getConvergedReason()
    ksp_converged = snes.getKSP().getConvergedReason()
    if snes_converged < 1 or ksp_converged < 1:
        PETSc.Sys.Print(f"SNES converged reason: {snes_converged}")
        PETSc.Sys.Print(f"KSP converged reason: {ksp_converged}")

    # Compute error
    def l2_err(u, uh):
        return mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form((uh - u) ** 2 * ufl.dx))**0.5,
            op=MPI.SUM)

    def h1_err(u, uh):
        return mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form(ufl.inner(
                    ufl.grad(uh - u), ufl.grad(uh - u)) * ufl.dx))**0.5,
            op=MPI.SUM)

    errorl2[j] = l2_err(u_soln, u)
    errorh1[j] = h1_err(u_soln, u)
    errorpl2[j] = l2_err(p_soln, p)
    errorph1[j] = h1_err(p_soln, p)

    h = dolfinx.cpp.mesh.h(
        mesh, 2, np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32))
    hsizes[j] = mesh.comm.allreduce(h.max(), op=MPI.MAX)


# Output convergence rates
if mesh.comm.rank == 0:
    hrates = np.log(hsizes[:-1] / hsizes[1:])
    print(f"L2 u convergence rates: "
          f"{np.log(errorl2[:-1] / errorl2[1:]) / hrates}")
    print(f"H1 u convergence rates: "
          f"{np.log(errorh1[:-1] / errorh1[1:]) / hrates}")
    print(f"L2 p convergence rates: "
          f"{np.log(errorpl2[:-1] / errorpl2[1:]) / hrates}")
    print(f"H1 p convergence rates: "
          f"{np.log(errorph1[:-1] / errorph1[1:]) / hrates}")
