import enum
import functools

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx

import dolfin_dg.dolfinx
import dolfin_dg.primal


class Compressible(enum.Enum):
    euler = enum.auto()
    navier_stokes = enum.auto()

def pprint(*msg, verbose=False):
    if verbose:
        PETSc.Sys.Print(", ".join(map(str, msg)))


ele_ns = [8, 16, 32]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

run_count = 0
print(f"Running problem ID: p={p}")

model = Compressible.navier_stokes
l = 0.5 * np.pi if model is Compressible.euler else np.pi
for ele_n in ele_ns:
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [l, l]],
        [ele_n, ele_n],
        cell_type=dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=dolfinx.mesh.DiagonalType.left)
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)
    h = ufl.CellDiameter(mesh)
    alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

    # -- Compressible Euler
    V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p), dim=4)
    v = ufl.TestFunction(V)

    U = dolfinx.fem.Function(V, name="u")
    U_soln = ufl.as_vector((ufl.sin(2 * (x[0] + x[1])) + 4,
                            0.2 * ufl.sin(2*(x[0]+x[1])) + 4,
                            0.2 * ufl.sin(2*(x[0]+x[1])) + 4,
                            (ufl.sin(2*(x[0]+x[1])) + 4) ** 2))
    U.interpolate(
        dolfinx.fem.Expression(U_soln, V.element.interpolation_points()))
    gamma = 1.4

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_1(_, flux):
        rho, u, E = dolfin_dg.aero.flow_variables(flux)
        p = dolfin_dg.aero.pressure(flux, gamma=gamma)
        H = dolfin_dg.aero.enthalpy(flux, gamma=gamma)

        inertia = rho * ufl.outer(u, u) + p * ufl.Identity(2)
        res = ufl.as_tensor([rho * u,
                             *[inertia[d, :] for d in range(2)],
                             rho * H * u])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return flux


    F_vec = [F_0, F_1]
    L_vec = [ufl.div]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, U, v)
    f = fos.F_vec[0](U_soln)
    F = fos.domain()

    # -- flux jacobian eigenvalues
    def generate_lambdas(U):
        rho, u, E = dolfin_dg.aero.flow_variables(U)
        p = dolfin_dg.aero.pressure(U, gamma=gamma)
        c = dolfin_dg.aero.speed_of_sound(p, rho, gamma=gamma)
        lambdas = [ufl.dot(u, n) - c, ufl.dot(u, n), ufl.dot(u, n) + c]
        return lambdas

    lambdas = generate_lambdas(U)
    lambdas_ext = generate_lambdas(U_soln)

    # -- interior
    eigen_vals_max_p = functools.reduce(
        dolfin_dg.math.max_value, (abs(l)("+") for l in lambdas))
    eigen_vals_max_m = functools.reduce(
        dolfin_dg.math.max_value, (abs(l)("-") for l in lambdas))
    alpha = dolfin_dg.math.max_value(
        eigen_vals_max_p, eigen_vals_max_m) / 2.0
    F += fos.interior([-alpha])

    # -- exterior
    eigen_vals_max_p = functools.reduce(
        dolfin_dg.math.max_value, map(abs, lambdas))
    eigen_vals_max_m = functools.reduce(
        dolfin_dg.math.max_value, map(abs, lambdas_ext))
    alpha = dolfin_dg.math.max_value(
        eigen_vals_max_p, eigen_vals_max_m) / 2.0
    F += fos.exterior([-alpha], U_soln)

    if model is Compressible.navier_stokes:
        mu = 1
        Pr = 0.72
        @dolfin_dg.primal.first_order_flux(lambda x: x)
        def F_ns_2(u, flux):
            return flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_ns_2(x)))
        def F_ns_1(u, flux):
            rho, rhou, rhoE = dolfin_dg.aero.conserved_variables(U)
            u = rhou/rho

            grad_rho = flux[0, :]
            grad_rhou = ufl.as_tensor([flux[j, :] for j in range(1, 3)])
            grad_rhoE = flux[3, :]

            # Quotient rule to find grad(u) and grad(E)
            grad_u = (grad_rhou*rho - ufl.outer(rhou, grad_rho))/rho**2
            grad_E = (grad_rhoE*rho - rhoE*grad_rho)/rho**2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(ufl.tr(grad_u))*ufl.Identity(2))
            K_grad_T = mu*gamma/Pr*(grad_E - ufl.dot(u, grad_u))

            res = ufl.as_tensor([ufl.zero(2),
                                 *(tau[d, :] for d in range(2)),
                                 tau * u + K_grad_T])
            return res

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_ns_1(x)))
        def F_ns_0(u, flux):
            return -flux


        F_vec = [F_ns_0, F_ns_1, F_ns_2]
        L_vec = [ufl.div, ufl.grad]

        fos_ns = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, U, v)
        F += fos_ns.domain()

        penalty = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h
        F += fos_ns.interior([penalty("+") * ufl.avg(fos_ns.G[1])])
        F += fos_ns.exterior([penalty * ufl.replace(fos_ns.G[1], {U: U_soln})],
                             U_soln)

        f += fos_ns.F_vec[0](U_soln)

    # -- source
    F += - ufl.inner(f, v) * ufl.dx

    # -- solve system
    du = ufl.TrialFunction(V)
    J = ufl.derivative(F, U, du)

    F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, U, [])

    total_dofs = mesh.comm.allreduce(
        V.dofmap.index_map.size_local * V.dofmap.index_map_bs, MPI.SUM)
    if mesh.comm.rank == 0:
        pprint(f"Solving problem: Nele={ele_n}, total DoFs = {total_dofs}")

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
    snes.setTolerances(rtol=1e-14, atol=1e-14)

    snes.solve(None, U.vector)
    if mesh.comm.rank == 0:
        pprint(f"SNES converged: {snes.getConvergedReason()}")
        pprint(f"KSP converged: {snes.getKSP().getConvergedReason()}")

    l2error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((U - U_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)
    errorl2[run_count] = l2error_u

    h1error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form(ufl.grad(U - U_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)
    errorh1[run_count] = h1error_u

    h_measure = dolfinx.cpp.mesh.h(
        mesh._cpp_object, 2,
        np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hsizes[run_count] = hmin

    run_count += 1

if mesh.comm.rank == 0:
    h_rates = np.log(hsizes[:-1] / hsizes[1:])
    print(f"L2 errors: {errorl2}")
    print(f"L2 error rates: {np.log(errorl2[:-1]/errorl2[1:]) / h_rates}")
    print(f"H1 error rates: {np.log(errorh1[:-1]/errorh1[1:]) / h_rates}")