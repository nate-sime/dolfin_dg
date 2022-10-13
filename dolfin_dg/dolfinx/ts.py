import abc
import enum

import dolfinx
import ufl
from petsc4py import PETSc


class PETScTSScheme(enum.Enum):
    implicit=1
    explicit=2
    imex=3
    implicit2=4


class PETScTSProblem(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def F(self, u, u_t, u_tt):
        pass

    def dFdu(self, u, u_t, u_tt):
        return ufl.derivative(self.F(u, u_t, u_tt), u)

    def dFdut(self, u, u_t, u_tt):
        return ufl.derivative(self.F(u, u_t, u_tt), u_t)

    def dFdutt(self, u, u_t, u_tt):
        return ufl.derivative(self.F(u, u_t, u_tt), u_tt)

    def G(self, u):
        pass

    def Gu(self, u):
        return ufl.derivative(self.G(u), u)


class PETScTSSolver:
    # F(t, u, u') = G(t, u)
    #
    # stiff
    #   F(t, u, u')   = (u', v) - f(t, u)
    #   F_u(t, u, u') = σ (u, v) - f_u(t, u)
    #   G(t, u) = 0
    #
    # stiff & non-stiff
    #   F(t, u, u')   = (u', v) - f(t, u)
    #   F_u(t, u, u') = σ (u, v) - f_u(t, u)
    #   G(t, u) = δc where (u, v) = δc g(t, u)

    def __init__(self, ts_problem, u, bcs, tdexprs=[],
                 mass_solver_prefix="mass_"):
        self.u = u
        self.V = u.function_space
        self.comm = self.V.mesh.comm
        self.tdexprs = tdexprs
        self.mass_solver_prefix = mass_solver_prefix

        # Vector used for rhs assembly
        # self.x0 = PETScVector(self.comm)

        # System `stiff' component, f(t, u)
        # self.F = PETScVector(self.comm)    # stiff vector

        self.u_t = dolfinx.fem.Function(self.V)
        self.u_tt = dolfinx.fem.Function(self.V)
        self.sigma_t = dolfinx.fem.Constant(self.V.mesh, 0.0)
        self.sigma_tt = dolfinx.fem.Constant(self.V.mesh, 0.0)

        du = ufl.TrialFunction(self.V)

        f = ts_problem.F(self.u, self.u_t, self.u_tt)
        f_j = self.sigma_tt*ts_problem.dFdutt(self.u, self.u_t, self.u_tt) + \
              self.sigma_t*ts_problem.dFdut(self.u, self.u_t, self.u_tt) + \
              ts_problem.dFdu(self.u, self.u_t, self.u_tt)
        # self.FJ = PETScMatrix(self.comm)   # stiff Jacobian

        # if not hasattr(bcs, "__len__"):
        #     bcs = (bcs,)
        # for bc in bcs:
        #     bc.apply(self.u.vector())
        self.bcs = bcs
        # self.f_assembler = SystemAssembler(f_j, f, self.bcs)
        # self.f_assembler.keep_diagonal = True

        # Non stiff terms (if provided)
        g = ts_problem.G(self.u)
        if g is not None:
            # self.G = PETScVector(self.comm)
            # self.GJ = PETScMatrix(self.comm)
            # self.M = PETScMatrix(self.comm)
            g_j = derivative(g, u)
            self.g_assembler = SystemAssembler(g_j, g, self.bcs)
            self.g_mass_solver = PETScKrylovSolver()
            # Typically, DG mass solver will use bjacobi
            self.g_mass_solver.set_options_prefix(self.mass_solver_prefix)
            self.g_mass_solver.set_operator(self.M)

        # TS object
        self.ts = PETSc.TS().create(self.comm)

    def evalIFunction(self, ts, t, x, xdot, F):
        # PETScTS implicit callback
        for tdexpr in self.tdexprs:
            tdexpr.t = t
        self.u.vector()[:] = PETScVector(x)
        self.u_t.vector()[:] = PETScVector(xdot)
        self.f_assembler.assemble(PETScVector(F), self.u.vector())

    def evalIJacobian(self, ts, t, x, xdot, sigma_t, A, P):
        # PETScTS implicit callback
        for tdexpr in self.tdexprs:
            tdexpr.t = t
        self.u.vector()[:] = PETScVector(x)
        self.u_t.vector()[:] = PETScVector(xdot)
        self.sigma_t.assign(sigma_t)
        self.f_assembler.assemble(PETScMatrix(A))

    def evalI2Function(self, ts, t, x, xdot, xdotdot, F):
        # PETScTS implicit callback
        for tdexpr in self.tdexprs:
            tdexpr.t = t
        self.u.vector()[:] = PETScVector(x)
        self.u_t.vector()[:] = PETScVector(xdot)
        self.u_tt.vector()[:] = PETScVector(xdotdot)
        self.f_assembler.assemble(PETScVector(F), self.u.vector())

    def evalI2Jacobian(self, ts, t, x, xdot, xdotdot, sigma_t, sigma_tt, A, P):
        # PETScTS implicit callback
        for tdexpr in self.tdexprs:
            tdexpr.t = t
        self.u.vector()[:] = PETScVector(x)
        self.u_t.vector()[:] = PETScVector(xdot)
        self.u_tt.vector()[:] = PETScVector(xdotdot)
        self.sigma_t.assign(sigma_t)
        self.sigma_tt.assign(sigma_tt)
        self.f_assembler.assemble(PETScMatrix(A))

    def evalRHSFunction(self, ts, t, x, F):
        # PETScTS explicit callback
        for tdexpr in self.tdexprs:
            tdexpr.t = t
        self.u.vector()[:] = PETScVector(x)
        self.g_assembler.assemble(PETScVector(F), self.u.vector())
        # self.g_mass_solver.solve(PETScVector(F), self.x0)

    def evalRHSJacobian(self, ts, t, x, A, P):
        # PETScTS explicit callback
        for tdexpr in self.tdexprs:
            tdexpr.t = t
        self.u.vector()[:] = PETScVector(x)
        self.g_assembler.assemble(PETScMatrix(A))

    def set_from_options(self):
        self.ts.setFromOptions()
        self.ts.getSNES().setFromOptions()

    def solve(self, t0=0.0, dt=0.1, max_t=1.0, max_steps=10,
              scheme=PETScTSScheme.implicit):
        # Initialise tensors with correct sparsity pattern
        self.f_assembler.assemble(self.FJ, self.F)

        # Solution vector
        w = Function(self.V)
        w.vector()[:] = self.u.vector()

        if scheme is PETScTSScheme.implicit:
            self.ts.setIFunction(self.evalIFunction, self.F.vec())
            self.ts.setIJacobian(self.evalIJacobian, J=self.FJ.mat(), P=self.FJ.mat())
            self.ts.setSolution(w.vector().vec())
        if scheme is PETScTSScheme.explicit:
            self.ts.setRHSFunction(self.evalRHSFunction, self.G.vec())
            self.ts.setRHSJacobian(self.evalRHSJacobian, J=self.GJ.mat(), P=self.GJ.mat())
            self.ts.setSolution(w.vector().vec())
        elif scheme is PETScTSScheme.imex:
            self.ts.setIFunction(self.evalIFunction, self.F.vec())
            self.ts.setIJacobian(self.evalIJacobian, J=self.FJ.mat(), P=self.FJ.mat())
            self.ts.setRHSFunction(self.evalRHSFunction, self.G.vec())
            self.ts.setRHSJacobian(self.evalRHSJacobian, J=self.GJ.mat(), P=self.GJ.mat())
            self.ts.setSolution(w.vector().vec())
        elif scheme is PETScTSScheme.implicit2:
            self.ts.setI2Function(self.evalI2Function, self.F.vec())
            self.ts.setI2Jacobian(self.evalI2Jacobian, J=self.FJ.mat(), P=self.FJ.mat())
            wdot = Function(self.V)
            self.ts.setSolution2(w.vector().vec(), wdot.vector().vec())

        # if self.g is not None:
        #     self.g_mass_solver.set_from_options()
        #     self.g_assembler.assemble(self.GJ, self.G)
        #
        #     self.ts.setRHSFunction(self.evalRHSFunction, f=self.G.vec())
        #     self.ts.setRHSJacobian(self.evalRHSJacobian, J=self.GJ.mat(), P=self.GJ.mat())

        self.ts.setTime(t0)
        self.ts.setTimeStep(dt)
        self.ts.setMaxTime(max_t)
        self.ts.setMaxSteps(max_steps)

        self.ts.solve(w.vector().vec())

        self.u.vector()[:] = w.vector()
