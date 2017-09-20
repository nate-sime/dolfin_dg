import ufl
from ufl import dot, Max

__author__ = 'njcs4'


class ConvectiveFlux:

    def __init__(self):
        pass

    def setup(self):
        pass

    def interior(self, F_c, u, n):
        pass

    def exterior(self, F_c, u, n):
        pass


class LocalLaxFriedrichs:

    def __init__(self, alpha_func):
        self.alpha_func = alpha_func
        self.alpha = None

    def setup(self, F_c, u, n):
        self.alpha = self.max_abs_eigenvalues(self.alpha_func(u, n))

    def interior(self, F_c, u_p, u_m, n):
        return 0.5*(dot(F_c(u_p), n) + dot(F_c(u_m), n) + self.alpha*(u_p - u_m))

    def exterior(self, F_c, u_p, u_m, n):
        return 0.5*(dot(F_c(u_p), n) + dot(F_c(u_m), n) + self.alpha*(u_p - u_m))

    def max_abs_eigenvalues(self, lambdas):
        if isinstance(lambdas, ufl.core.expr.Expr):
            return abs(lambdas)

        assert isinstance(lambdas, (list, tuple))

        lambdas = list(map(abs, lambdas))
        alpha = Max(lambdas[0], lambdas[1])
        for j in range(2, len(lambdas)):
            alpha = Max(alpha, lambdas[j])
        return alpha


def lax_friedrichs_flux(F_c, alpha):
    def H(U_p, U_m, n_p):
        return 0.5*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + alpha*(U_p - U_m))
    return H