import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx

import dolfin_dg.dolfinx
from dolfin_dg.math import hyper_tensor_T_product as G_T_mult
from dolfin_dg.primal.facet_sipg import DivIBP, GradIBP, CurlIBP
from dolfin_dg.math import homogenize


def pprint(*msg, verbose=False):
    if verbose:
        print(msg)


ele_ns = [8, 16, 32]
# ele_ns = [2, 4]
errorl2_u = np.zeros(len(ele_ns))
errorh1_u = np.zeros(len(ele_ns))
errorl2_vel = np.zeros(len(ele_ns))
errorh1_vel = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
dofs_counter = np.zeros(len(ele_ns))
p = 2

formulation = 1

run_count = 0
for ele_n in ele_ns:
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array((-1.0, -1.0)), np.array((1.0, 1.0))],
        [ele_n, ele_n],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=dolfinx.mesh.DiagonalType.left)
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    # mu = dolfinx.fem.Constant(mesh, 1.0)
    mu = (1 + x[0]**2 * x[1]**2)

    vel_soln = ufl.as_vector((
        2*x[1]*(1.0 - x[0]*x[0]),
        -2*x[0]*(1.0 - x[1]*x[1])))
    Vs = dolfinx.fem.FunctionSpace(mesh, ("DG", p + 1))
    dofs_counter[run_count] = mesh.comm.allreduce(Vs.dofmap.index_map.size_local, MPI.SUM)

    # Compute the approximation of the true solution of u
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim-1,
        marker=lambda x: np.ones_like(x[0], dtype=np.int8))
    dofs = dolfinx.fem.locate_dofs_topological(
        Vs, mesh.topology.dim-1, facets)
    zero_bc = dolfinx.fem.dirichletbc(0.0, dofs, Vs)

    u_soln = dolfinx.fem.Function(Vs)
    vs = ufl.TestFunction(Vs)
    F_psi = ufl.dot(ufl.grad(u_soln), ufl.grad(vs)) * ufl.dx - ufl.curl(vel_soln) * vs * ufl.dx

    problem = dolfinx.fem.petsc.NonlinearProblem(F_psi, u_soln, [zero_bc])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    opts = PETSc.Options()
    option_prefix = solver.krylov_solver.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    solver.krylov_solver.setFromOptions()

    solver.solve(u_soln)

    # -- Biharmonic
    V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
    v = ufl.TestFunction(V)

    u = dolfinx.fem.Function(V, name="u")

    if formulation == 1:
        def F_4(u, flux=None):
            if flux is None:
                flux = u
            return flux

        def F_3(u, flux=None):
            if flux is None:
                flux = ufl.grad(F_4(u))
            return flux

        def F_2(u, flux=None):
            if flux is None:
                flux = ufl.div(F_3(u))
            return mu * flux

        def F_1(u, flux=None):
            if flux is None:
                flux = ufl.grad(F_2(u))
            return flux

        def F_0(u, flux=None):
            if flux is None:
                flux = ufl.div(F_1(u))
            return flux

        # f = F_0(u_soln)
        f = ufl.curl(-ufl.div(mu * ufl.grad(vel_soln)))

        # Homogenisers
        G0 = homogenize(F_0, u, ufl.div(F_1(u)))
        G1 = homogenize(F_1, u, ufl.grad(F_2(u)))
        G2 = homogenize(F_2, u, ufl.div(F_3(u)))
        G3 = homogenize(F_3, u, ufl.grad(F_4(u)))

        # Domain
        F = ufl.inner(F_2(u), ufl.div(G_T_mult(G1, ufl.grad(G_T_mult(G0, v))))) * ufl.dx \
            - ufl.inner(f, v) * ufl.dx

        # Interior
        # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
        h = ufl.CellDiameter(mesh)
        alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
        beta = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

        # F0(u, F1(u)) = G0 div(F1(u))
        divibp = DivIBP(F_1, u, v, G0)
        F += divibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
        F += divibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}), u, u_soln, u_soln)

        # F1(u) = G1 grad(F2(u))
        gradibp = GradIBP(F_2, u, ufl.grad(G_T_mult(G0, v)), G1)
        F -= gradibp.interior_residual1(beta("+") * ufl.avg(G2), ufl.grad(u))
        F -= gradibp.exterior_residual1(
            beta * ufl.replace(G2, {u: u_soln}), ufl.grad(u), ufl.grad(u_soln), u_soln)

        # F2(u, F3(u)) = G2 div(F3(u))
        divibp = DivIBP(F_3, u, ufl.div(G_T_mult(G1, ufl.grad(G_T_mult(G0, v)))), G2)
        F += divibp.interior_residual2()
        F += divibp.exterior_residual2(u_soln)

        # F3(u) = G3 grad(F4(u))
        gradibp = GradIBP(F_4, u, ufl.grad(G_T_mult(G2, ufl.div(G_T_mult(G1, ufl.grad(G_T_mult(G0, v)))))), G3)
        F -= gradibp.interior_residual2()
        F -= gradibp.exterior_residual2(u_soln)
    elif formulation == 2:

        def F_4(u, flux=None):
            if flux is None:
                flux = u
            return flux


        def F_3(u, flux=None):
            if flux is None:
                flux = ufl.curl(F_4(u))
            return flux


        def F_2(u, flux=None):
            if flux is None:
                flux = ufl.grad(F_3(u))
            return mu * (flux + flux.T)


        def F_1(u, flux=None):
            if flux is None:
                flux = ufl.div(F_2(u))
            return flux


        def F_0(u, flux=None):
            if flux is None:
                flux = ufl.curl(F_1(u))
            return -flux


        # f = F_0(u_soln)
        f = ufl.curl(-ufl.div(mu*(ufl.grad(vel_soln) + ufl.grad(vel_soln).T)))

        # Homogenisers
        G0 = homogenize(F_0, u, ufl.curl(F_1(u)))
        G1 = homogenize(F_1, u, ufl.div(F_2(u)))
        G2 = homogenize(F_2, u, ufl.grad(F_3(u)))
        G3 = homogenize(F_3, u, ufl.curl(F_4(u)))

        # Domain
        F = - ufl.inner(F_2(u), ufl.grad(
            G_T_mult(G1, ufl.curl(G_T_mult(G0, v))))) * ufl.dx \
            - ufl.inner(f, v) * ufl.dx

        # Interior
        # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
        h = ufl.CellDiameter(mesh)
        alpha = dolfinx.fem.Constant(mesh,
                                     10.0 * p ** (4 if p <= 2 else 6)) / h ** 3
        beta = dolfinx.fem.Constant(mesh, 10.0 * p ** 2) / h

        # F0(u, F1(u)) = G0 curl(F1(u))
        curl1ibp = CurlIBP(F_1, u, v, G0)
        F += curl1ibp.interior_residual1(alpha("+") * ufl.avg(G1), u)
        F += curl1ibp.exterior_residual1(alpha * ufl.replace(G1, {u: u_soln}),
                                         u, u_soln, u_soln)

        # F1(u) = G1 div(F2(u))
        divibp = DivIBP(F_2, u, ufl.curl(G_T_mult(G0, v)), G1)
        F += divibp.interior_residual1(beta("+") * ufl.avg(G2), ufl.curl(u))
        F += divibp.exterior_residual1(
            beta * ufl.replace(G2, {u: u_soln}), ufl.curl(u), ufl.curl(u_soln),
            u_soln)

        # F2(u, F3(u)) = G2 grad(F3(u))
        gradibp = GradIBP(F_3, u,
                          ufl.grad(G_T_mult(G1, ufl.curl(G_T_mult(G0, v)))), G2)
        F -= gradibp.interior_residual2()
        F -= gradibp.exterior_residual2(u_soln)

        # F3(u) = G3 curl(F4(u))
        curl2ibp = CurlIBP(F_4, u, ufl.div(
            G_T_mult(G2, ufl.grad(G_T_mult(G1, ufl.curl(G_T_mult(G0, v)))))),
                           G3)
        F += curl2ibp.interior_residual2()
        F += curl2ibp.exterior_residual2(u_soln)


    du = ufl.TrialFunction(V)
    J = ufl.derivative(F, u, du)

    F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])

    total_dofs = mesh.comm.allreduce(
        V.dofmap.index_map.size_local * V.dofmap.index_map_bs, MPI.SUM)
    if mesh.comm.rank == 0:
        pprint(f"Solving problem: Nele={ele_n}, total DoFs = {total_dofs}")

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    # opts["snes_monitor"] = None
    # opts["snes_max_it"] = 1
    snes.setFromOptions()
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
    snes.setTolerances(rtol=1e-14, atol=1e-14)

    snes.solve(None, u.vector)
    if mesh.comm.rank == 0:
        pprint(f"SNES converged: {snes.getConvergedReason()}")
        pprint(f"KSP converged: {snes.getKSP().getConvergedReason()}")

    l2error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorl2_u[run_count] = l2error_u

    h1error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form(ufl.grad(u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorh1_u[run_count] = h1error_u

    h_measure = dolfinx.cpp.mesh.h(
        mesh, 2, np.arange(mesh.topology.connectivity(2, 0).num_nodes,
                           dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hsizes[run_count] = hmin

    # Interpolate the streamfunction in the (p-1) velocity space
    VEL = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", p-1))
    vel_expr = dolfinx.fem.Expression(
        ufl.curl(u), VEL.element.interpolation_points())
    vel = dolfinx.fem.Function(VEL)
    vel.interpolate(vel_expr)

    l2error_vel = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((vel - vel_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorl2_vel[run_count] = l2error_vel

    h1error_vel = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form(ufl.grad(vel - vel_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorh1_vel[run_count] = h1error_vel

    run_count += 1

if mesh.comm.rank == 0:
    h_rates = np.log(hsizes[:-1] / hsizes[1:])
    print(f"p = {p}")
    print(f"DoFs: {dofs_counter}")

    print(f"phi L2 errors: {errorl2_u}")
    print(f"phi L2 error rates: {np.log(errorl2_u[:-1] / errorl2_u[1:]) / h_rates}")
    print(f"phi H1 error rates: {np.log(errorh1_u[:-1] / errorh1_u[1:]) / h_rates}")

    print(f"u L2 errors: {errorl2_vel}")
    print(f"u L2 error rates: {np.log(errorl2_vel[:-1] / errorl2_vel[1:]) / h_rates}")
    print(f"u H1 error rates: {np.log(errorh1_vel[:-1] / errorh1_vel[1:]) / h_rates}")