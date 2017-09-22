import ufl
from ufl import dot, Max, Min

__author__ = 'njcs4'


def max_abs_of_sequence(a):
    """
    Utility function to generate the Max Abs of a sequence of N elements
    e.g.: Max(|a1|, |a2|, |a3|, ..., |aN|).

    This is required because (currently) ufl only allows two values in
    the constuctor of Max().

    :param a: sequence of ufl elements
    :return:
    """
    if isinstance(a, ufl.core.expr.Expr):
        return abs(a)

    assert isinstance(a, (list, tuple))

    a = list(map(abs, a))
    return max_of_sequence(a)


def max_of_sequence(a):
    """
    Utility function to generate the Max of a sequence of N elements
    e.g.: Max(a1, a2, a3, ..., aN).

    This is required because (currently) ufl only allows two values in
    the constuctor of Max().

    :param a: sequence of ufl elements
    :return:
    """
    return map_ufl_operator_to_sequence(a, Max)


def min_of_sequence(a):
    """
    Utility function to generate the Max of a sequence of N elements
    e.g.: Max(a1, a2, a3, ..., aN).

    This is required because (currently) ufl only allows two values in
    the constuctor of Max().

    :param a: sequence of ufl elements
    :return:
    """
    return map_ufl_operator_to_sequence(a, Min)


def map_ufl_operator_to_sequence(a, op):
    """
    Utility function to generate the op of a sequence of N elements
    e.g.: op(a1, a2, a3, ..., aN).

    This is required because (currently and commonly) ufl only allows
    two values in the constuctors of MathOperator.

    Warning: If the sequence has length 1, op(a) is assumed == a

    This is intended to be used with Min and Max

    :param a: sequence of ufl elements
    :param op: a ufl MathOperator
    :return:
    """
    if isinstance(a, ufl.core.expr.Expr):
        return a

    assert isinstance(a, (list, tuple))

    alpha = op(a[0], a[1])
    for j in range(2, len(a)):
        alpha = op(alpha, a[j])
    return alpha


class ConvectiveFlux:

    def __init__(self):
        pass

    def setup(self):
        pass

    def interior(self, F_c, u_p, u_m, n):
        pass

    def exterior(self, F_c, u_p, u_m, n):
        pass


class LocalLaxFriedrichs(ConvectiveFlux):

    def __init__(self, flux_jacobian_eigenvalues):
        self.flux_jacobian_eigenvalues = flux_jacobian_eigenvalues
        self.alpha = None

    def setup(self, F_c, u_p, u_m, n):
        eigen_vals_max_p = max_abs_of_sequence(self.flux_jacobian_eigenvalues(u_p, n))
        eigen_vals_max_m = max_abs_of_sequence(self.flux_jacobian_eigenvalues(u_m, n))
        self.alpha = Max(eigen_vals_max_p, eigen_vals_max_m)

    def interior(self, F_c, u_p, u_m, n):
        return 0.5*(dot(F_c(u_p), n) + dot(F_c(u_m), n) + self.alpha*(u_p - u_m))

    exterior = interior


class HLLE(ConvectiveFlux):

    TOL = 1e-14

    def __init__(self, flux_jacobian_eigenvalues):
        self.flux_jacobian_eigenvalues = flux_jacobian_eigenvalues

    def setup(self, F_c, u_p, u_m, n):
        u_avg = (u_p + u_m)/2

        eigen_vals_max_p = max_of_sequence(self.flux_jacobian_eigenvalues(u_avg, n))
        eigen_vals_min_m = min_of_sequence(self.flux_jacobian_eigenvalues(u_avg, n))
        self.lam_p = Max(eigen_vals_max_p, 0)
        self.lam_m = Min(eigen_vals_min_m, 0)

    def interior(self, F_c, u_p, u_m, n):
        lam_p, lam_m = self.lam_p, self.lam_m
        guard = ufl.conditional(abs(lam_p - lam_m) < HLLE.TOL, 0, 1/(lam_p - lam_m))
        return guard*(lam_p*dot(F_c(u_p), n) - lam_m*dot(F_c(u_m), n) - lam_p*lam_m*(u_p - u_m))

    exterior = interior


class Vijayasundaram(ConvectiveFlux):

    def __init__(self, eigenvals, left, right):
        self.eigenvals = eigenvals
        self.left = left
        self.right = right

    def setup(self, F_c, u_p, u_m, n):
        pass

    def interior(self, F_c, u_p, u_m, n):
        u_avg = (u_p + u_m)/2
        avg_eigs = self.eigenvals(u_avg, n)

        if isinstance(avg_eigs, (list, tuple)):
            m = len(avg_eigs)
            max_ev = ufl.as_matrix([[Max(avg_eigs[i], 0) if i == j else 0 for i in range(m)] for j in range(m)])
            min_ev = ufl.as_matrix([[Min(avg_eigs[j], 0) if i == j else 0 for i in range(m)] for j in range(m)])
        else:
            max_ev = Max(avg_eigs, 0)
            min_ev = Min(avg_eigs, 0)

        left_vecs = self.left(u_avg, n)
        right_vecs = self.right(u_avg, n)

        B_p = right_vecs*max_ev*left_vecs
        B_m = right_vecs*min_ev*left_vecs

        return B_p*u_p + B_m*u_m

    exterior = interior