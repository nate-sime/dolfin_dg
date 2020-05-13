from petsc4py import PETSc
import numpy as np
import ufl
import dolfinx
import dolfin_dg
import dolfin_dg.dolfinx
import numpy
import numba
from mpi4py import MPI

comm = MPI.COMM_WORLD


def u_soln_f(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])


n_eles = [8, 16, 32]
l2errors_u = np.zeros_like(n_eles, dtype=np.double)
l2errors_p = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = dolfinx.UnitSquareMesh(comm, n_ele, n_ele)
    Ve_high = ufl.FiniteElement("CG", mesh.ufl_cell(), 4)
    Ve = ufl.FiniteElement("DG", mesh.ufl_cell(), 1)
    Vbare = ufl.FiniteElement("DGT", mesh.ufl_cell(), 1)

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

    alpha = dolfinx.Constant(mesh, 1.0)
    h = ufl.CellDiameter(mesh)
    b = dolfinx.Constant(mesh, (1, 1))

    f = -ufl.div(ufl.grad(u_soln)) + ufl.div(b*u_soln)

    n = ufl.FacetNormal(mesh)
    zeta = ufl.conditional(ufl.ge(ufl.dot(b, n), 0), 0, 1)

    def facet_integral(integrand):
        return integrand('-') * ufl.dS + integrand('+') * ufl.dS + integrand * ufl.ds

    # Second order terms
    F = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        - f * v * ufl.dx

    F += - facet_integral((u - ubar)*ufl.dot(ufl.grad(v), n)) \
         + facet_integral(ufl.dot(-ufl.grad(u) - alpha/h * n * (ubar - u), n) * v)

    F -= facet_integral(ufl.dot(-ufl.grad(u) - alpha/h * n * (ubar - u), n) * vbar)

    # First order terms
    F += - ufl.inner(b*u, ufl.grad(v)) * ufl.dx
    F += facet_integral(ufl.inner((1 - zeta) * ufl.dot(b*u, n) + zeta * ufl.dot(b*ubar, n), v))
    F -= facet_integral(ufl.inner((1 - zeta) * ufl.dot(b*u, n) + zeta * ufl.dot(b*ubar, n), vbar))

    J = ufl.derivative(F, U)

    facets = dolfinx.mesh.locate_entities_boundary(mesh, 1,
                                      lambda x: np.logical_or.reduce((
                                          np.isclose(x[0], 0.0),
                                          np.isclose(x[0], 1.0),
                                          np.isclose(x[1], 0.0),
                                          np.isclose(x[1], 1.0)
                                      )))
    facet_dofs = dolfinx.fem.locate_dofs_topological((W.sub(1), Vbar), 1, facets)
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
