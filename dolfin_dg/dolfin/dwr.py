import ufl
from dolfin import (
    FunctionSpace, Function, TrialFunction, TestFunction, solve,
    PETScKrylovSolver, PETScMatrix, PETScVector,
    assemble_system, info, derivative, MeshFunction, cells, DirichletBC,
    assemble
)

from dolfin_dg.dolfin.mark import Marker


def dual(form, w, z=None):
    if z is not None:
        v, u = form.arguments()
        return ufl.replace(form, {u: w, v: z})
    u = form.arguments()[0]
    return ufl.replace(form, {u: w})


class NonlinearAPosterioriEstimator:

    def __init__(self, J, F, j, u_h, bcs=None, p_inc=1, options_prefix=None):
        assert(len(F.arguments()) == 1)
        assert(len(J.arguments()) == 2)
        assert(len(j.arguments()) == 0)

        if bcs is None:
            bcs = []

        if not hasattr(bcs, "__len__"):
            bcs = (bcs,)

        self.options_prefix = options_prefix

        V = F.arguments()[0].function_space()
        e = V.ufl_element()
        self.V_star = FunctionSpace(
            V.mesh(), e.reconstruct(degree=e.degree()+p_inc))

        # Copy and homogenise BCs
        for bc in bcs:
            if bc.user_sub_domain() is None:
                raise NotImplementedError(
                    "BCs defined by mesh functions not yet supported")

        self.bcs = []
        for bc in bcs:
            dwr_subspc = self.V_star
            for i in bc.function_space().component():
                dwr_subspc = dwr_subspc.sub(i)
            self.bcs += [DirichletBC(dwr_subspc, bc.value(),
                                     bc.user_sub_domain())]

        for bc in self.bcs:
            bc.homogenize()

        self.F = F
        self.J = J
        self.j = j
        self.u_h = u_h

        info("NonlinearAPosterioriEstimator space and dual space size: (%d, %d)"
             % (V.dim(), self.V_star.dim()))

    def compute_cell_markers(self, marker):
        assert(isinstance(marker, Marker))
        eta_k = self.compute_indicators()
        return marker.mark(eta_k)

    def compute_indicators(self):
        z = self.compute_dual_solution()
        eta = self.compute_error_indicators(z)
        eta_cf = self.compute_cell_indicators(eta)
        return eta_cf

    def compute_dual_solution(self):
        # Replace the components of the bilinear and linear formulations to
        # formulate the dual problem
        # u -> w, v -> z
        w = TestFunction(self.V_star)
        z = TrialFunction(self.V_star)

        dual_M = dual(self.J, w, z)
        dual_j = derivative(self.j, self.u_h, w)

        # Solve the dual problem
        z_s = Function(self.V_star)

        if self.options_prefix is None:
            solve(dual_M == dual_j, z_s, self.bcs,
                  solver_parameters={'linear_solver': 'mumps'})
        else:
            dM, dj = PETScMatrix(), PETScVector()
            assemble_system(dual_M, dual_j, self.bcs, A_tensor=dM, b_tensor=dj)
            linear_solver = PETScKrylovSolver()
            linear_solver.set_options_prefix(self.options_prefix)
            linear_solver.set_from_options()
            linear_solver.set_operator(dM)
            linear_solver.solve(z_s.vector(), dj)

        return z_s

    def compute_error_indicators(self, z):
        # Evaluate the residual in the enriched space
        v = self.F.arguments()[0]

        DG0 = FunctionSpace(self.V_star.mesh(), "DG", 0)
        dg_0 = TestFunction(DG0)

        dwr = ufl.replace(self.F, {v: z*dg_0})

        dwr_vec = assemble(dwr)
        dwr_vec.abs()
        eta = Function(DG0, dwr_vec)

        return eta

    def compute_cell_indicators(self, eta):
        mesh = self.V_star.mesh()

        # Put the values of the projection into a cell function
        cf = MeshFunction("double", mesh, mesh.topology().dim(), 0.0)
        for c in cells(mesh):
            cf[c] = eta.vector()[c.index()]
        return cf


class LinearAPosterioriEstimator(NonlinearAPosterioriEstimator):

    def __init__(self, a, L, j, u_h, bcs=None, p_inc=1):
        assert(len(L.arguments()) == 1)
        assert(len(a.arguments()) == 2)
        assert(len(j.arguments()) == 1)

        F = a - L
        v, u = F.arguments()
        F = ufl.replace(F, {u: u_h})
        J = derivative(F, u_h)

        u = j.arguments()[0]
        j = ufl.replace(j, {u: u_h})

        NonlinearAPosterioriEstimator.__init__(
            self, J, F, j, u_h, bcs=bcs, p_inc=p_inc)
