import numpy as np
import ufl

import dolfinx
import dolfinx.plotting

import dolfin_dg
import dolfin_dg.dolfinx

from petsc4py import PETSc
from mpi4py import MPI

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


for matrixtype in list(dolfin_dg.dolfinx.MatrixType):
    l2errors_u = []
    l2errors_p = []
    hs = []

    for run_no, n in enumerate([8, 16, 32]):
        mesh = dolfinx.UnitSquareMesh(
            comm, n, n,
            cell_type=dolfinx.cpp.mesh.CellType.triangle,
            ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

        # Higher order FE spaces for interpolation of the true solution
        V_high = dolfinx.VectorFunctionSpace(mesh, ("DG", p_order + 2))
        Q_high = dolfinx.FunctionSpace(mesh, ("DG", p_order + 1))

        u_soln = dolfinx.Function(V_high)
        u_soln.interpolate(u_analytical)

        p_soln = dolfinx.Function(Q_high)
        p_soln.interpolate(p_analytical)

        # Problem FE spaces and FE functions
        Ve = ufl.VectorElement("DG", mesh.ufl_cell(), p_order)
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), p_order-1)

        if not matrixtype.is_block_type():
            W = dolfinx.FunctionSpace(mesh, ufl.MixedElement([Ve, Qe]))
            U = dolfinx.Function(W)
            u, p = ufl.split(U)
            dU = ufl.TrialFunction(W)
            VV = ufl.TestFunction(W)
            v, q = ufl.split(VV)
        else:
            V = dolfinx.FunctionSpace(mesh, Ve)
            Q = dolfinx.FunctionSpace(mesh, Qe)
            u, p = dolfinx.Function(V), dolfinx.Function(Q)
            U = [u, p]
            du, dp = ufl.TrialFunction(V), ufl.TrialFunction(Q)
            v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

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
        facets = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim - 1,
                                       bc_facet_ids, bc_idxs)
        ds = ufl.Measure("ds", subdomain_data=facets)
        dsD, dsN = ds(dirichlet_id), ds(neumann_id)

        # Stokes system nonlinear viscosity model and viscous flux
        def eta(u):
            return 1 + 1e-1 * ufl.dot(u, u)

        def F_v(u, grad_u, p_local=None):
            if p_local is None:
                p_local = p
            return 2*eta(u)*ufl.sym(grad_u) - p_local*ufl.Identity(
                mesh.geometry.dim)

        # Formulate weak BCs
        facet_n = ufl.FacetNormal(mesh)
        gN = F_v(u_soln, ufl.grad(u_soln), p_soln)*facet_n
        bcs = [dolfin_dg.DGDirichletBC(dsD, u_soln),
               dolfin_dg.DGNeumannBC(dsN, gN)]

        stokes = dolfin_dg.StokesOperator(mesh, None, bcs, F_v)

        # Residual, Jacobian and preconditioner FE formulations
        F = stokes.generate_fem_formulation(
            u, v, p, q, block_form=matrixtype.is_block_type())

        f = -ufl.div(F_v(u_soln, ufl.grad(u_soln), p_soln))
        if not matrixtype.is_block_type():
            F -= ufl.inner(f, v) * ufl.dx
        else:
            F[0] -= ufl.inner(f, v) * ufl.dx

        J = dolfin_dg.derivative_block(F, U)

        P = None
        if matrixtype.is_block_type():
            P = [[J[0][0], J[0][1]],
                 [J[1][0], (2 * eta(u))**-1 * dp * q * ufl.dx]]

        # Setup SNES solver
        snes = PETSc.SNES().create(comm)
        opts = PETSc.Options()
        opts["snes_monitor"] = None
        snes.setFromOptions()

        # Setup nonlinear problem
        problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(
            F, J, U, [], P=P)

        # Construct linear system data structures
        if matrixtype is dolfin_dg.dolfinx.MatrixType.monolithic:
            snes.setFunction(problem.F_mono, dolfinx.fem.create_vector(F))
            snes.setJacobian(problem.J_mono, J=dolfinx.fem.create_matrix(J),
                             P=None)
            soln_vector = U.vector
        elif matrixtype is dolfin_dg.dolfinx.MatrixType.block:
            snes.setFunction(problem.F_block,
                             dolfinx.fem.create_vector_block(F))
            snes.setJacobian(problem.J_block,
                             J=dolfinx.fem.create_matrix_block(J),
                             P=None)
            soln_vector = dolfinx.fem.create_vector_block(F)

            # Copy initial guess into vector
            dolfinx.cpp.la.scatter_local_vectors(
                soln_vector, [u.vector.array_r, p.vector.array_r],
                [u.function_space.dofmap.index_map,
                 p.function_space.dofmap.index_map])
        elif matrixtype is dolfin_dg.dolfinx.MatrixType.nest:
            snes.setFunction(problem.F_nest, dolfinx.fem.create_vector_nest(F))
            snes.setJacobian(problem.J_nest,
                             J=dolfinx.fem.create_matrix_nest(J),
                             P=dolfinx.fem.create_matrix_nest(P))
            soln_vector = dolfinx.fem.create_vector_nest(F)

            # Copy initial guess into vector
            for soln_vec_sub, var_sub in zip(soln_vector.getNestSubVecs(), U):
                soln_vec_sub.array[:] = var_sub.vector.array_r[:]
                soln_vec_sub.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                         mode=PETSc.ScatterMode.REVERSE)

        # Set solver options
        if matrixtype is not dolfin_dg.dolfinx.MatrixType.nest:
            snes.getKSP().getPC().setType("lu")
            snes.getKSP().getPC().setFactorSolverType("mumps")
        else:
            nested_IS = snes.getJacobian()[0].getNestISs()
            snes.getKSP().setType("fgmres")
            snes.getKSP().setTolerances(rtol=1e-12)
            snes.getKSP().getPC().setType("fieldsplit")
            snes.getKSP().getPC().setFieldSplitIS(
                ["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
            snes.getKSP().getPC().setFieldSplitType(
                PETSc.PC.CompositeType.ADDITIVE)

            ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
            ksp_u.setType("preonly")
            ksp_u.getPC().setType("hypre")
            ksp_p.setType("preonly")
            ksp_p.getPC().setType("hypre")

        # Solve and check convergence
        snes.solve(None, soln_vector)
        snes_converged = snes.getConvergedReason()
        ksp_converged = snes.getKSP().getConvergedReason()
        if snes_converged < 1 or ksp_converged < 1:
            print("SNES converged reason:", snes_converged)
            print("KSP converged reason:", ksp_converged)

        # Computer error
        l2error_u = comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                (u - u_soln) ** 2 * ufl.dx)**0.5,
            op=MPI.SUM)
        l2error_p = comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                (p - p_soln) ** 2 * ufl.dx)**0.5,
            op=MPI.SUM)

        hs.append(comm.allreduce(mesh.hmin(), op=MPI.MIN))
        l2errors_u.append(l2error_u)
        l2errors_p.append(l2error_p)

    # Compute convergence rates
    l2errors_u = np.array(l2errors_u)
    l2errors_p = np.array(l2errors_p)
    hs = np.array(hs)

    hrates = np.log(hs[:-1] / hs[1:])
    rates_u = np.log(l2errors_u[:-1] / l2errors_u[1:]) / hrates
    rates_p = np.log(l2errors_p[:-1] / l2errors_p[1:]) / hrates
    print(matrixtype, "rates u: %s" % str(rates_u))
    print(matrixtype, "rates p: %s" % str(rates_p))
