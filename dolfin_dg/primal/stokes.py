import typing
import ufl

import dolfin_dg.primal


def stokes_continuity(
        u: ufl.core.expr.Expr,
        q: ufl.core.expr.Expr):

    # -- mass
    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_mass_1(U, flux):
        return U

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_mass_1(x)))
    def F_mass_0(U, flux):
        return flux

    fos = dolfin_dg.primal.FirstOrderSystem(
        [F_mass_0, F_mass_1], [ufl.div], u, q, n_ibps=[2])
    return fos


def stokes_stress(
        u: ufl.core.expr.Expr,
        v: ufl.core.expr.Expr,
        p: ufl.core.expr.Expr,
        eta: ufl.core.expr.Expr):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the Stokes
    stress oeprator:

    .. math :: - \nabla \cdot (2 \eta \varepsilon (u) - p I )

    where :math:`A` is a positive definite tensor
    """
    dim = u.ufl_shape[0]

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(U, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(U, flux):
        return 2 * eta * ufl.sym(flux) - p * ufl.Identity(dim)

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(U, flux):
        return -flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos
