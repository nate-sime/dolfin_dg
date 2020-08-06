import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfin_dg.dolfinx
import dolfin_dg.hdg_form

comm = MPI.COMM_WORLD


def u_soln_f(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
    # return np.exp(x[0] - x[1])


poly_o = 2
n_eles = [8, 16, 32]
l2errors_u = np.zeros_like(n_eles, dtype=np.double)
l2errors_p = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = dolfinx.UnitSquareMesh(
        comm, n_ele, n_ele, ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
    Ve_high = ufl.FiniteElement("CG", mesh.ufl_cell(), poly_o+2)
    Ve = ufl.FiniteElement("DG", mesh.ufl_cell(), poly_o)
    Vbare = ufl.FiniteElement("DGT", mesh.ufl_cell(), poly_o)

    W = dolfinx.FunctionSpace(mesh, ufl.MixedElement([Ve, Vbare]))
    V = W.sub(0).collapse()
    Vbar = W.sub(1).collapse()

    U = dolfinx.Function(W)
    v_test = ufl.TestFunction(W)

    u, ubar = ufl.split(U)
    v, vbar = ufl.split(v_test)

    u_soln = dolfinx.Function(dolfinx.FunctionSpace(mesh, Ve_high))
    u_soln.interpolate(u_soln_f)

    gD = dolfinx.Function(V)
    gDbar = dolfinx.Function(Vbar)
    gD.interpolate(u_soln_f)
    gDbar.interpolate(u_soln_f)

    alpha = dolfinx.Constant(mesh, 10.0 * Ve.degree()**2)
    h = ufl.CellDiameter(mesh)
    b = dolfinx.Constant(mesh, (1, 1))

    n = ufl.FacetNormal(mesh)

    # Second order terms
    kappa = dolfinx.Constant(mesh, 2.0)

    def F_v(u, grad_u):
        return (kappa + u**2) * grad_u

    F = ufl.inner(F_v(u, ufl.grad(u)), ufl.grad(v)) * ufl.dx

    sigma = alpha / h
    G = dolfin_dg.homogeneity_tensor(F_v, u)
    hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
        F_v, u, ubar, v, vbar, sigma, G, n)

    F += hdg_term.face_residual(ufl.dS, ufl.ds)

    # First order terms
    def F_c(u):
        return b * u ** 2

    F += -ufl.inner(F_c(u), ufl.grad(v)) * ufl.dx

    H_flux = dolfin_dg.LocalLaxFriedrichs(
        flux_jacobian_eigenvalues=lambda u, n: 2 * u * ufl.dot(b, n))
    hdg_fo_term = dolfin_dg.hdg_form.HDGClassicalFirstOrder(
        F_c, u, ubar, v, vbar, H_flux, n)

    F += hdg_fo_term.face_residual(ufl.dS, ufl.ds)

    # Volume source
    f = ufl.div(F_c(u_soln) - F_v(u_soln, ufl.grad(u_soln)))
    F += - f * v * ufl.dx

    J = ufl.derivative(F, U)

    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        )))
    facet_dofs = dolfinx.fem.locate_dofs_topological((
        W.sub(1), Vbar), 1, facets)
    bc = dolfinx.DirichletBC(gDbar, facet_dofs, W.sub(1))

    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, U, [bc])

    snes = PETSc.SNES().create(mesh.mpi_comm())
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.setFunction(problem.F_mono, dolfinx.fem.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.create_matrix(J), P=None)
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.solve(None, U.vector)

    l2error_u = comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar((u - u_soln) ** 2 * ufl.dx) ** 0.5,
        op=MPI.SUM)

    hs[run_no] = comm.allreduce(mesh.hmin(), op=MPI.MIN)
    l2errors_u[run_no] = l2error_u

print(l2errors_u)
rates_u = np.log(l2errors_u[:-1] / l2errors_u[1:]) / np.log(hs[:-1] / hs[1:])
print("rates u: %s" % str(rates_u))
