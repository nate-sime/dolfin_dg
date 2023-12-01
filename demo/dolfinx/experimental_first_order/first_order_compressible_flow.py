import enum
import functools

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc

import dolfin_dg.dolfinx
import dolfin_dg.primal
import dolfin_dg.primal.aero
import dolfin_dg.primal.flux_sipg_ext_avg
import dolfin_dg.penalty


class Compressible(enum.Enum):
    euler = enum.auto()
    euler_entropy = enum.auto()
    navier_stokes = enum.auto()
    navier_stokes_entropy = enum.auto()


def pprint(*msg):
    PETSc.Sys.Print(", ".join(map(str, msg)))


for model in Compressible:
    pprint(f"*** Running model: {model}")

    ele_ns = [8, 16, 32]
    errorl2 = np.zeros(len(ele_ns))
    errorh1 = np.zeros(len(ele_ns))
    hsizes = np.zeros(len(ele_ns))
    p = 1

    gamma = 1.4
    run_count = 0

    l = np.pi
    if model in (Compressible.euler, Compressible.euler_entropy):
        l = 0.5 * np.pi

    metadata = {}
    if model in (Compressible.euler_entropy, Compressible.navier_stokes_entropy):
        metadata["quadrature_degree"] = 2*p + 5
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", metadata=metadata)
    dS = ufl.Measure("dS", metadata=metadata)

    for ele_n in ele_ns:
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            [[0.0, 0.0], [l, l]],
            [ele_n, ele_n],
            cell_type=dolfinx.mesh.CellType.triangle,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            diagonal=dolfinx.mesh.DiagonalType.right)

        # -- Compressible Euler
        V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p), dim=4)
        v = ufl.TestFunction(V)

        soln_vec = dolfinx.fem.Function(V, name="u")
        x = ufl.SpatialCoordinate(mesh)
        U_soln = ufl.as_vector((ufl.sin(2 * (x[0] + x[1])) + 4,
                                0.2 * ufl.sin(2*(x[0]+x[1])) + 4,
                                0.2 * ufl.sin(2*(x[0]+x[1])) + 4,
                                (ufl.sin(2*(x[0]+x[1])) + 4) ** 2))

        if model in (Compressible.euler_entropy,
                     Compressible.navier_stokes_entropy):
            gD = dolfin_dg.primal.aero.U_to_V(U_soln, gamma)
            U = dolfin_dg.primal.aero.V_to_U(soln_vec, gamma)
            fos = dolfin_dg.primal.aero.compressible_euler_entropy(soln_vec, v, gamma=gamma)
        else:
            gD = U_soln
            U = soln_vec
            fos = dolfin_dg.primal.aero.compressible_euler(soln_vec, v, gamma=gamma)

        soln_vec.interpolate(
            dolfinx.fem.Expression(gD, V.element.interpolation_points()))

        f = fos.F_vec[0](gD)
        F = fos.domain(dx=dx)

        # -- flux jacobian eigenvalues
        rho, u, E = dolfin_dg.aero.flow_variables(U)
        pressure = dolfin_dg.aero.pressure(U, gamma=gamma)
        c = dolfin_dg.aero.speed_of_sound(pressure, rho, gamma=gamma)
        n = ufl.FacetNormal(mesh)
        lambdas = [ufl.dot(u, n) - c, ufl.dot(u, n), ufl.dot(u, n) + c]
        alpha, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
            lambdas, soln_vec, gD)

        F += fos.interior([-alpha], flux_type=dolfin_dg.primal.flux_sipg_ext_avg, dS=dS)
        F += fos.exterior([-alpha_ext], gD, flux_type=dolfin_dg.primal.flux_sipg_ext_avg, ds=ds)

        if model in (Compressible.navier_stokes,
                     Compressible.navier_stokes_entropy):
            mu = 1
            Pr = 0.72
            if model is Compressible.navier_stokes:
                fos_ns = dolfin_dg.primal.aero.compressible_navier_stokes(
                    soln_vec, v, gamma=gamma, mu=mu, Pr=Pr)
            elif model is Compressible.navier_stokes_entropy:
                fos_ns = dolfin_dg.primal.aero.compressible_navier_stokes_entropy(
                    soln_vec, v, gamma=gamma, mu=mu, Pr=Pr)

            F += fos_ns.domain(dx=dx)

            pen, pen_ext = dolfin_dg.penalty.interior_penalty(fos_ns, soln_vec, gD)
            F += fos_ns.interior([pen], dS=dS)
            F += fos_ns.exterior([pen_ext], gD, ds=ds)
            f += fos_ns.F_vec[0](gD)

        # -- source
        F += - ufl.inner(f, v) * dx

        # -- solve system
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, soln_vec, du)

        F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
        problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(
            F, J, soln_vec, [])

        total_dofs = mesh.comm.allreduce(
            V.dofmap.index_map.size_local * V.dofmap.index_map_bs, MPI.SUM)
        if mesh.comm.rank == 0:
            pprint(f"Solving problem: Nele={ele_n}, total DoFs = {total_dofs}")

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        opts = PETSc.Options()
        # opts["snes_monitor"] = None
        snes.setFromOptions()
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
        snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
        snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
        snes.setTolerances(rtol=1e-14, atol=1e-14)

        snes.solve(None, soln_vec.vector)
        if mesh.comm.rank == 0:
            snes_conv = snes.getConvergedReason()
            ksp_conv = snes.getKSP().getConvergedReason()
            if snes_conv <= 0 or ksp_conv <= 0:
                pprint(f"SNES converged: {snes.getConvergedReason()}")
                pprint(f"KSP converged: {snes.getKSP().getConvergedReason()}")

        l2error_u = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form((U - U_soln) ** 2 * dx)) ** 0.5,
            op=MPI.SUM)
        errorl2[run_count] = l2error_u

        h1error_u = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form(ufl.grad(U - U_soln) ** 2 * dx)) ** 0.5,
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