import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx

import dolfin_dg.dolfinx
from dolfin_dg import HLLE, HyperbolicOperator, DGDirichletBC

def G_T_mult(G, tau):
    if len(G.ufl_shape) == 0:
        if not len(tau.ufl_shape) == 0:
            raise IndexError("G^T is scalar, tau has shape: %s"
                             + str(tau.ufl_shape))
        return G*tau
    elif ufl.rank(tau) == 0 and ufl.rank(G) == 2:
        return G.T * tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return ufl.dot(G.T, tau)
    elif ufl.rank(tau) == 1:
        return ufl.dot(G.T, tau)
    m, d = tau.ufl_shape
    return ufl.as_matrix([[ufl.inner(G[:, :, i, k], tau) for k in range(d)]
                          for i in range(m)])


def tensor_jump(u, n):
    return ufl.outer(u, n)("+") + ufl.outer(u, n)("-")

# def G_T(G):
#     if len(G.ufl_shape) == 0:
#         return G
#     return G.T
#
#
# def G_mult(G, v):
#     if len(v.ufl_shape) == 0:
#         return G * v
#     return ufl.dot(G, v)

class IBP:

    def __init__(self, F, u, v, G):
        self.F = F
        self.u = u
        self.v = v
        self.G = G
        print(f"Initialising {self}")
        print(f"Shape F(u) = {F(u).ufl_shape}")
        print(f"Shape G = {G.ufl_shape}")
        print(f"Shape u = {u.ufl_shape}")
        print(f"Shape v = {v.ufl_shape}")


class DivIBP(IBP):

    def interior_residual1(self, alpha, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.function_space)
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        print(f"Div Shape (ufl.avg(F(u)), tensor_jump(G_T_v, n)): {ufl.avg(F(u)).ufl_shape, tensor_jump(G_T_v, n).ufl_shape}")
        print(f"Div Shape tensor_jump(G_T_v, n), tensor_jump(u, n): {tensor_jump(G_T_v, n).ufl_shape, tensor_jump(u, n).ufl_shape}")
        # quit()
        R = ufl.inner(ufl.avg(F(u)), tensor_jump(G_T_v, n)) * dS \
            - ufl.inner(alpha * tensor_jump(G_T_v, n), tensor_jump(u, n)) * dS
        return R

    def exterior_residual1(self, alpha, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.function_space)
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)
        R = ufl.inner(0.5*(F(u) + F(uD)), ufl.outer(G_gamma_T_v, n)) * ds \
            - ufl.inner(alpha * ufl.outer(G_gamma_T_v, n), ufl.outer(u - uD, n)) * ds
        return R


class GradIBP(IBP):

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.function_space)
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        print(
            f"Grad Shape ufl.avg(G_T_v), tensor_jump(F(u), n): {ufl.avg(G_T_v).ufl_shape, tensor_jump(F(u), n).ufl_shape})")

        R = -ufl.inner(ufl.avg(G_T_v), tensor_jump(F(u), n)) * dS
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.function_space)
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)
        # Should this be avg(G)?
        R = -ufl.inner(G_gamma_T_v, ufl.outer(F(u) - F(uD), n)) * ds
        return R


def homogenize(F, diff_op):
    diff_op = ufl.variable(diff_op)
    G = ufl.diff(F(u, diff_op), diff_op)
    return G


run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 2

for ele_n in ele_ns:
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, ele_n, ele_n,
        cell_type=dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=dolfinx.mesh.DiagonalType.right)
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    problem = 4
    if problem == 1:
        V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
        v = ufl.TestFunction(V)

        u = dolfinx.fem.Function(V, name="u")
        u.interpolate(lambda x: x[0] + 1)

        u_soln = ufl.exp(x[0] - x[1])
        # f = dolfinx.fem.Constant(mesh, np.array(0, dtype=np.double))
        b = dolfinx.fem.Constant(mesh, np.array((1, 1), dtype=np.double))

        # Convective Operator
        def F_c(U, flux=None):
            if flux is None:
                flux = b*U
            return flux

        f = ufl.div(F_c(u_soln))

        # Domain
        F = - ufl.inner(F_c(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

        # Interior
        eigen_vals_max_p = abs(ufl.dot(ufl.diff(F_c(u), u), n)("+"))
        eigen_vals_max_m = abs(ufl.dot(ufl.diff(F_c(u), u), n)("-"))
        alpha = ufl.Max(eigen_vals_max_p, eigen_vals_max_m) / 2.0

        G = ufl.as_ufl(1)
        divibp = DivIBP(F_c, u, v, G)
        F += divibp.interior_residual1(-alpha)

        # Exterior
        alpha = abs(ufl.dot(b, n)) / 2.0
        F += divibp.exterior_residual1(-alpha, u_soln)
    elif problem == 2:
        V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
        v = ufl.TestFunction(V)

        u = dolfinx.fem.Function(V, name="u")
        u.interpolate(lambda x: x[0] + 1)

        u_soln = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]) + 1

        # Convective Operator
        def F_2(u, flux=None):
            if flux is None:
                flux = u
            return flux

        def F_1(u, flux=None):
            if flux is None:
                flux = ufl.grad(F_2(u))
            return (u + 1) * flux

        def F_0(u, flux=None):
            if flux is None:
                flux = ufl.div(F_1(u))
            return -flux

        # f = F_0(u_soln, F_1(u_soln, F_2(u_soln, u_soln)))
        f = F_0(u_soln, ufl.div(F_1(u_soln)))

        # Domain
        F = ufl.inner(F_1(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

        # Interior
        # h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
        h = ufl.CellDiameter(mesh)
        alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

        # F0(u, F1(u)) = -div(F1(u))
        G0 = homogenize(F_0, ufl.div(F_1(u)))
        divibp = DivIBP(F_1, u, v, G0)

        F += divibp.interior_residual1(alpha("+"))
        F += divibp.exterior_residual1(alpha, u_soln)

        # F1(u) = G1 grad(F2(u))
        G1 = homogenize(F_1, ufl.grad(u))
        gradibp = GradIBP(F_2, u, ufl.grad(v), G1)
        F += gradibp.interior_residual2()
        F += gradibp.exterior_residual2(u_soln)
    elif problem == 3:
        V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
        v = ufl.TestFunction(V)

        u = dolfinx.fem.Function(V, name="u")
        u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1.0)))

        u_soln = ufl.as_vector(
            [ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]) + 1]*2)

        # Convective Operator
        def F_2(u, flux=None):
            if flux is None:
                flux = u
            return flux

        def F_1(u, flux=None):
            if flux is None:
                flux = ufl.grad(F_2(u))
            return flux

        def F_0(u, flux=None):
            if flux is None:
                flux = ufl.div(F_1(u))
            return -flux

        # f = F_0(u_soln, F_1(u_soln, F_2(u_soln, u_soln)))
        f = F_0(u_soln, ufl.div(F_1(u_soln)))

        # Domain
        F = ufl.inner(F_1(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

        # Interior
        h = ufl.CellDiameter(mesh)
        alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h

        # F0(u, F1(u)) = -div(F1(u))
        G0 = homogenize(F_0, ufl.div(F_1(u)))
        divibp = DivIBP(F_1, u, v, G0)
        F += divibp.interior_residual1(alpha("+"))
        F += divibp.exterior_residual1(alpha, u_soln)

        # F1(u) = G1 grad(F2(u))
        G1 = homogenize(F_1, ufl.grad(u))
        gradibp = GradIBP(F_2, u, ufl.grad(v), G1)
        F += gradibp.interior_residual2()
        F += gradibp.exterior_residual2(u_soln)
    elif problem == 4:
        V = dolfinx.fem.VectorFunctionSpace(mesh, ('DG', p))
        v = ufl.TestFunction(V)

        u = dolfinx.fem.Function(V, name="u")
        u.interpolate(lambda x: np.stack((x[0] + 1, x[0] + 1.0)))
        # u_soln = dolfinx.fem.Function(V)
        # u_soln.interpolate(lambda x: np.stack([np.zeros_like(x[0])]*2))
        # u_soln = ufl.as_vector((x[0], x[1]))
        u_soln = ufl.as_vector(
            [ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])]*2)

        E = 10.0
        nu = 0.3
        mu = dolfinx.fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
        lmda = dolfinx.fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

        # Convective Operator
        def F_2(u, flux=None):
            if flux is None:
                flux = u
            return flux

        def F_1(u, grad_u=None):
            if grad_u is None:
                grad_u = ufl.grad(F_2(u))
            return 2 * mu * ufl.sym(grad_u) + lmda * ufl.tr(ufl.sym(grad_u)) * ufl.Identity(2)

        def F_0(u, flux=None):
            if flux is None:
                flux = ufl.div(F_1(u))
            return -flux

        # f = F_0(u_soln, F_1(u_soln, F_2(u_soln, u_soln)))
        f = F_0(u_soln, ufl.div(F_1(u_soln)))

        # Domain
        F = ufl.inner(F_1(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

        # Interior
        h = ufl.CellDiameter(mesh)
        alpha = dolfinx.fem.Constant(mesh, 200.0) * p**2 / h

        # F0(u, F1(u)) = -div(F1(u))
        G0 = homogenize(F_0, ufl.div(F_1(u)))
        divibp = DivIBP(F_1, u, v, G0)
        F += divibp.interior_residual1(alpha("+"))
        F += divibp.exterior_residual1(alpha, u_soln)

        # F1(u) = G1 grad(F2(u))
        G1 = homogenize(F_1, ufl.grad(u))
        gradibp = GradIBP(F_2, u, ufl.grad(v), G1)
        F += gradibp.interior_residual2()
        F += gradibp.exterior_residual2(u_soln)

    du = ufl.TrialFunction(V)
    J = ufl.derivative(F, u, du)

    F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
    snes.setTolerances(rtol=1e-14, atol=1e-14)

    snes.solve(None, u.vector)
    print(f"SNES converged: {snes.getConvergedReason()}")
    print(f"KSP converged: {snes.getKSP().getConvergedReason()}")

    # from dolfinx.io import VTXWriter
    # with VTXWriter(mesh.comm, "output.bp", u) as f:
    #     f.write(0.0)
    # # u.interpolate(lambda x: np.exp(x[0] - x[1]))
    # u.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
    # with VTXWriter(mesh.comm, "soln.bp", u) as f:
    #     f.write(0.0)
    # quit()
    # import febug
    # febug.plot_function(u).show()

    l2error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorl2[run_count] = l2error_u

    h1error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form(ufl.grad(u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorh1[run_count] = h1error_u

    h_measure = dolfinx.cpp.mesh.h(
        mesh, 2, np.arange(mesh.topology.connectivity(2, 0).num_nodes,
                           dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hsizes[run_count] = hmin

    run_count += 1

if mesh.comm.rank == 0:
    h_rates = np.log(hsizes[:-1] / hsizes[1:])
    print(f"L2 errors: {errorl2}")
    print(f"L2 error rates: {np.log(errorl2[:-1]/errorl2[1:]) / h_rates}")
    print(f"H1 error rates: {np.log(errorh1[:-1]/errorh1[1:]) / h_rates}")