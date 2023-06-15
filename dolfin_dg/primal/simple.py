import ufl

import dolfin_dg.primal


def advection(u, v, b):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the advection
    oeprator:

    .. math :: \nabla \vec{b} u

    where :math:`\vec{b}` is a :math:`d` dimensional vector
    """

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_1(u, flux):
        return b * flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return flux

    F_vec = [F_0, F_1]
    L_vec = [ufl.div]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos


def diffusion(u, v, A):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the diffusion
    oeprator:

    .. math :: - \nabla \cdot A \nabla u

    where :math:`A` is a positive definite tensor
    """
    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(u, flux):
        return A * flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return -flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos


def maxwell(u, v):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the Maxwell
    problem:

    .. math :: \nabla \times \nabla \times \nabla u

    """

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.curl(F_2(x)))
    def F_1(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.curl(F_1(x)))
    def F_0(u, flux):
        return flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.curl, ufl.curl]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos


def biharmonic(u, v):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    biharmonic problem:

    .. math :: \nabla^4 u

    where

    .. math :: \varepsilon(\vec{v}) = \frac{1}{2} (\nabla \vec{v} + \nabla \vec{v}^\top)

    """

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_4(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_4(x)))
    def F_3(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_3(x)))
    def F_2(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return flux

    F_vec = [F_0, F_1, F_2, F_3, F_4]
    L_vec = [ufl.div, ufl.grad, ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos


def streamfunction(u, v, mu):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    streamfunction problem:

    .. math :: \nabla \times (-\nabla \cdot 2 mu \varepsilon( \nabla \times u ))

    where

    .. math :: \varepsilon(\vec{v}) = \frac{1}{2} (\nabla \vec{v} + \nabla \vec{v}^\top)

    """

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_4(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.curl(F_4(x)))
    def F_3(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_3(x)))
    def F_2(u, flux):
        return mu * (flux + flux.T)

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_2(x)))
    def F_1(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.curl(F_1(x)))
    def F_0(u, flux):
        return -flux

    F_vec = [F_0, F_1, F_2, F_3, F_4]
    L_vec = [ufl.curl, ufl.div, ufl.grad, ufl.curl]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos


def triharmonic(u, v):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    triharmonic problem:

    .. math :: \nabla^6 u

    """

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_6(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_6(x)))
    def F_5(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_5(x)))
    def F_4(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_4(x)))
    def F_3(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_3(x)))
    def F_2(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return flux

    F_vec = [F_0, F_1, F_2, F_3, F_4, F_5, F_6]
    L_vec = [ufl.div, ufl.grad, ufl.div, ufl.grad, ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
    return fos
