import ufl
from ufl import dot, Max, Min


def max_abs_of_sequence(a):
    r"""Utility function to generate the maximum of the absolute values of
    elements in a sequence

    .. math::

        \max(|a_1|, |a_2|, |a_3|, \ldots, |a_N|)

    Notes
    -----
    This is required because (currently) ufl only allows two values in
    the constuctor of :py:meth:`ufl.Max`.

    Parameters
    ----------
    a
        Sequence of ufl expressions

    Returns
    -------
    Maximum of absolute values of elements in the sequence
    """
    if isinstance(a, ufl.core.expr.Expr):
        return abs(a)

    assert isinstance(a, (list, tuple))

    a = list(map(abs, a))
    return max_of_sequence(a)


def max_of_sequence(a):
    r"""Utility function to generate the maximum of the absolute values of
    elements in a sequence

    .. math::

        \max(a_1, a_2, a_3, \ldots, a_N)

    Notes
    -----
    This is required because (currently) ufl only allows two values in
    the constuctor of :py:meth:`ufl.Max`.

    Returns
    -------
    Maximum value of elements in the sequence
    """
    return map_ufl_operator_to_sequence(a, Max)


def min_of_sequence(a):
    r"""Utility function to generate the maximum of the absolute values of
    elements in a sequence

    .. math::

        \min(a_1, a_2, a_3, \ldots, a_N)

    Notes
    -----
    This is required because (currently) ufl only allows two values in
    the constuctor of :py:meth:`ufl.Max`.

    Returns
    -------
    Minimum value of elements in the sequence
    """
    return map_ufl_operator_to_sequence(a, Min)


def map_ufl_operator_to_sequence(a, op):
    """
    Utility function to map an operator to a sequence of N UFL expressions

    Notes
    -----
    This is required because (currently and commonly) ufl only allows
    two values in the constuctors of :py:meth:`ufl.MathOperator`. This is
    intended to be used with :py:meth:`ufl.Min` and :py:mesh`ufl.Max`.
    """
    if isinstance(a, ufl.core.expr.Expr):
        return a

    assert isinstance(a, (list, tuple))

    alpha = op(a[0], a[1])
    for j in range(2, len(a)):
        alpha = op(alpha, a[j])
    return alpha


class ConvectiveFlux:
    """Abstract bass class of symbolic convection flux approximations
    """

    def __init__(self):
        pass

    def setup(self):
        """Called internally prior to formulation by
        :py:meth:`ConvectiveFlux.interior` and
        :py:meth:`ConvectiveFlux.exterior`.
        """
        pass

    def interior(self, F_c, u_p, u_m, n):
        """Formulate interior residual formulation using the symbolic flux
        formulation

        Parameters
        ----------
        F_c
            Convective flux tensor
        u_p
            ":math:`+`" outward flux solution vector
        u_m
            ":math:`-`" inward flux solution vector
        n
            ":math:`+`" outward side facet normal

        Returns
        -------
        Interior residual formulation
        """
        pass

    def exterior(self, F_c, u_p, u_m, n):
        """Formulate exterior residual formulation using the symbolic flux
        formulation

        Parameters
        ----------
        F_c
            Convective flux tensor
        u_p
            ":math:`+`" outward flux solution vector
        u_m
            ":math:`-`" inward flux function or solution vector
        n
            Facet normal (outward pointing)

        Returns
        -------
        Exterior residual formulation
        """
        pass


class LocalLaxFriedrichs(ConvectiveFlux):
    """Implementation of symbolic representation of the local-Lax Friedrichs
    flux function
    """

    def __init__(self, flux_jacobian_eigenvalues):
        self.flux_jacobian_eigenvalues = flux_jacobian_eigenvalues
        self.alpha = None

    def setup(self, F_c, u_p, u_m, n):
        eigen_vals_max_p = max_abs_of_sequence(
            self.flux_jacobian_eigenvalues(u_p, n))
        eigen_vals_max_m = max_abs_of_sequence(
            self.flux_jacobian_eigenvalues(u_m, n))
        self.alpha = Max(eigen_vals_max_p, eigen_vals_max_m)

    def interior(self, F_c, u_p, u_m, n):
        return 0.5*(dot(F_c(u_p), n) + dot(F_c(u_m), n)
                    + self.alpha*(u_p - u_m))

    exterior = interior


class ModifiedLocalLaxFriedrichs(LocalLaxFriedrichs):
    """Implementation of symbolic representation of the space-time local-Lax Friedrichs
    flux function
    """

    def __init__(self, flux_jacobian_eigenvalues):
        super(self.__class__, self).__init__(flux_jacobian_eigenvalues)

    def interior(self, F_c, u_p, u_m, n):
        a = (len(n)-1)*[self.alpha]
        A = ufl.as_tensor([*a,1.])

        alpha = abs(ufl.dot(A,n))

        return 0.5*(ufl.dot(F_c(u_p), n) + ufl.dot(F_c(u_m), n)
                    + alpha*(u_p - u_m))

    exterior = interior


class HLLE(ConvectiveFlux):
    """Implementation of the Harten-Lax-van Leer-Einfeldt flux function
    """

    TOL = 1e-14

    def __init__(self, flux_jacobian_eigenvalues):
        self.flux_jacobian_eigenvalues = flux_jacobian_eigenvalues

    def setup(self, F_c, u_p, u_m, n):
        u_avg = (u_p + u_m)/2

        eigen_vals_max_p = max_of_sequence(
            self.flux_jacobian_eigenvalues(u_avg, n))
        eigen_vals_min_m = min_of_sequence(
            self.flux_jacobian_eigenvalues(u_avg, n))
        self.lam_p = Max(eigen_vals_max_p, 0)
        self.lam_m = Min(eigen_vals_min_m, 0)

    def interior(self, F_c, u_p, u_m, n):
        lam_p, lam_m = self.lam_p, self.lam_m
        guard = ufl.conditional(
            abs(lam_p - lam_m) < HLLE.TOL, 0, 1/(lam_p - lam_m))
        return guard*(lam_p*dot(F_c(u_p), n)
                      - lam_m*dot(F_c(u_m), n)
                      - lam_p*lam_m*(u_p - u_m))

    exterior = interior


class Vijayasundaram(ConvectiveFlux):
    """Implementation of the Vijayasundaram flux function
    """

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
            max_ev = ufl.as_matrix(
                [[Max(avg_eigs[i], 0) if i == j else 0 for i in range(m)]
                 for j in range(m)])
            min_ev = ufl.as_matrix(
                [[Min(avg_eigs[j], 0) if i == j else 0 for i in range(m)]
                 for j in range(m)])
        else:
            max_ev = Max(avg_eigs, 0)
            min_ev = Min(avg_eigs, 0)

        left_vecs = self.left(u_avg, n)
        right_vecs = self.right(u_avg, n)

        B_p = right_vecs*max_ev*left_vecs
        B_m = right_vecs*min_ev*left_vecs

        return B_p*u_p + B_m*u_m

    exterior = interior
