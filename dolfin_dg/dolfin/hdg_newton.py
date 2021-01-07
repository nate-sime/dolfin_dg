import leopart
import dolfin


def _formit(F):
    if isinstance(F, (list, tuple)):
        return list(map(_formit, F))
    return dolfin.Form(F)


class StaticCondensationNewtonSolver:

    def __init__(self, F, J, bcs,
                 rtol=1e-12, atol=1e-10, maximum_iterations=20,
                 krylov_solver=None):
        """
        Newton solver which implements static condensation via `leopart`. The
        residual and Jacobian are constructed as follows:

        F = [F_local, F_global]
        J = [[J_(local,local),  J_(local,global) ],
             [J_(global,local), J_(global,global)]]

        where "local" refers to components of the variational formulation
        whose unknown components are defined cellwise with no inter-cell
        communication (e.g. DG elements), and "global" refers to the
        components of the variational formulation (typically defined on
        facets in HDG) which must be solved in a global sense as a necessity of
        inter-entity communication.

        Parameters
        ----------
        F
            Iterable of size 2 containing the residual block formulation
        J
            Iterable of size 2x2 containing the block Jacobi formulation
        bcs
            Boundary conditions
        rtol
            Residual 2-norm residual tolerance
        atol
            Residual 2-norm absolute tolerance
        maximum_iterations
            Maximum number of Newton iterations
        krylov_solver
            Custom implementation of a `PETScKrylov` solver
        """
        if not hasattr(bcs, "__len__"):
            bcs = (bcs,)
        self.bcs = bcs
        self.bcs_homo = list(map(dolfin.DirichletBC, self.bcs))
        for bc in self.bcs_homo:
            bc.homogenize()

        J = _formit(J)
        F = _formit(F)
        self.assembler = leopart.AssemblerStaticCondensation(
            J[0][0], J[0][1],
            J[1][0], J[1][1],
            F[0], F[1],
            bcs
        )

        self.A = dolfin.PETScMatrix()
        self.b = dolfin.PETScVector()

        if krylov_solver is None:
            solver = dolfin.PETScKrylovSolver()
            solver.set_operator(self.A)
            dolfin.PETScOptions.set("ksp_type", "preonly")
            dolfin.PETScOptions.set("pc_type", "lu")
            dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
            solver.set_from_options()
            self.solver = solver
        else:
            self.solver = krylov_solver

        self.maximum_iterations = maximum_iterations
        self.rtol = rtol
        self.atol = atol

    def converged(self, r, it):
        _residual = r.norm("l2")

        if it == 0:
            self._residual0 = _residual

        if self._residual0 > self.atol:
            relative_residual = _residual / self._residual0
        else:
            relative_residual = 1.0

        dolfin.info(
            "(static condensation) Newton iteration %d: "
            "r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e)"
             % (it, _residual, self.atol, relative_residual, self.rtol))

        return relative_residual < self.rtol or _residual < self.atol

    def solve(self, u, ubar):
        for bc in self.bcs:
            if u.function_space().contains(
                    dolfin.FunctionSpace(bc.function_space())):
                bc.apply(u.vector())
            if ubar.function_space().contains(
                    dolfin.FunctionSpace(bc.function_space())):
                bc.apply(ubar.vector())

        du = u.copy(deepcopy=True)
        dubar = ubar.copy(deepcopy=True)

        self.assembler.assemble_global(self.A, self.b)
        for bc in self.bcs_homo:
            bc.apply(self.A, self.b)

        if self.converged(self.b, 0):
            return 0, True

        for newton_iteration in range(1, self.maximum_iterations+1):
            self.solver.solve(dubar.vector(), self.b)
            self.assembler.backsubstitute(
                dubar._cpp_object, du._cpp_object)

            u.vector().axpy(-1.0, du.vector())
            ubar.vector().axpy(-1.0, dubar.vector())

            self.assembler.assemble_global_rhs(self.b)
            for bc in self.bcs_homo:
                bc.apply(self.b)

            if self.converged(self.b, newton_iteration):
                return newton_iteration, True

            if newton_iteration == self.maximum_iterations:
                return newton_iteration, False

            self.assembler.assemble_global_lhs(self.A)
            for bc in self.bcs_homo:
                bc.apply(self.A)
