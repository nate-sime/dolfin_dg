import functools
import typing
import ufl
import ufl.algorithms.apply_derivatives

import dolfin_dg
import dolfin_dg.primal


def local_lax_friedrichs_penalty(
        lambdas: typing.Union[
            ufl.core.expr.Expr, typing.Sequence[ufl.core.expr.Expr]],
        u: ufl.core.expr.Expr,
        u_ext: ufl.core.expr.Expr):
    """
    Generate local Lax Friedrichs flux penalisation parameter :math:`\alpha`
    where

    .. math::

        \mathcal{H}_{\mathrm{LF}}(u^+, u^-, n) = \frac{1}{2} ( F^+ \cdot n + F^- \cdot n + \alpha (u^+ - u^-))

    Parameters
    ----------
    lambdas
        Flux Jacobian eigenvalues
    u
        Unknown numerical flux (finite element function)
    u_ext
        Known (or unknown) exterior flux

    Returns
    -------
    :math:`\alpha` defined on in the interior and exterior
    """
    if not isinstance(lambdas, (list, tuple)):
        lambdas = (lambdas,)

    lambdas = tuple(map(ufl.algorithms.apply_derivatives.apply_derivatives,
                        lambdas))

    def max_reduce(items):
        return functools.reduce(dolfin_dg.math.max_value, items)

    # -- interior
    eigen_vals_max_p = max_reduce((abs(l)("+") for l in lambdas))
    eigen_vals_max_m = max_reduce((abs(l)("-") for l in lambdas))
    alpha = dolfin_dg.math.max_value(eigen_vals_max_p, eigen_vals_max_m) / 2.0

    # -- exterior
    eigen_vals_max_p = max_reduce(map(abs, lambdas))
    lambdas_ext = list(map(lambda l: ufl.replace(l, {u: u_ext}), lambdas))
    eigen_vals_max_m = max_reduce(map(abs, lambdas_ext))
    alpha_ext = dolfin_dg.math.max_value(
        eigen_vals_max_p, eigen_vals_max_m) / 2.0

    return alpha, alpha_ext


def interior_penalty(fos: dolfin_dg.primal.FirstOrderSystem,
                     u: ufl.coefficient.Coefficient,
                     u_ext: ufl.core.expr.Expr,
                     c_ip: typing.Union[ufl.core.expr.Expr, float] = 20.0,
                     h_measure: typing.Optional[ufl.core.expr.Expr] = None):
    """
    Generate the interior penalty parameter as proposed in Hartmann & Houston
    2008:

    .. math::

        \delta = C_{\mathrm{IP}} \frac{p^2}{h} \{G(u)\}


    Parameters
    ----------
    fos
        The related :class:`dolfin_dg.primal.FirstOrderSystem`
    u
        FE function
    u_ext
        Exterior numerical flux
    c_ip
        Penalty parameter
    h_measure
        Measure of cell size

    Returns
    -------
    :math:`\delta` as evaluated on the interior and exterior
    """
    if h_measure is None:
        h_measure = dolfin_dg.primal.default_h_measure(u.ufl_domain())
    p = dolfin_dg.dg_form._get_ufl_element_degree(u)

    # TODO: This needs to account for the bounds on the C_{inv,p,q} constant
    #  in the trace inequalities.
    alpha = c_ip * p ** 2 / h_measure

    penalty = dolfin_dg.math.max_value(
        alpha("+"), alpha("-")) * ufl.avg(fos.G[1])

    penalty_ext = alpha * ufl.replace(fos.G[1], {u: u_ext})
    return penalty, penalty_ext
