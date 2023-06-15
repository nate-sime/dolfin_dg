import ufl

import dolfin_dg
import dolfin_dg.primal

def compressible_euler(U, v, gamma=1.4):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    compressible Euler operator:

    .. math ::
        \nabla \cdot
        \begin{pmatrix}
        \rho \vec{u} \\
        \rho \vec{u} \otimes \vec{u} + p I \\
        (\rho E + p) \vec{u}
        \end{pmatrix}

    where :math:`U` is the solution vector

    .. math::
        \mathbf{U} =
        \begin{pmatrix}
            \rho \\
            \rho u \\
            \rho E
        \end{pmatrix}

    and :math:`\gamma` is the ratio of specific heats.
    """
    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_1(_, flux):
        rho, u, E = dolfin_dg.aero.flow_variables(flux)
        p = dolfin_dg.aero.pressure(flux, gamma=gamma)
        H = dolfin_dg.aero.enthalpy(flux, gamma=gamma)

        inertia = rho * ufl.outer(u, u) + p * ufl.Identity(2)
        res = ufl.as_tensor([rho * u,
                             *[inertia[d, :] for d in range(2)],
                             rho * H * u])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return flux

    F_vec = [F_0, F_1]
    L_vec = [ufl.div]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, U, v)
    return fos


def compressible_navier_stokes(U, v, gamma=1.4, mu=1, Pr=0.72):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    compressible Navier Stokes operator:

    .. math ::
        -\nabla \cdot
        \begin{pmatrix}
        \vec{0} \\
        \sigma \\
        \sigma \vec{u} + \kappa \nabla T
        \end{pmatrix}

    where :math:`\sigma = \mu \left( \nabla \vec{u} + \nabla \vec{u}^\top
    - \frac{2}{3} (\nabla \cdot \vec{u}) I \right)`,

    :math:`U` is the solution vector

    .. math::
        \mathbf{U} =
        \begin{pmatrix}
            \rho \\
            \rho u \\
            \rho E
        \end{pmatrix},

    :math:`\gamma` is the ratio of specific heats, :math:`\mu` is the viscosity
    and :math:`\mathrm{Pr}` is the Prandtl number.

    Notes
    -----
    Combine this formulation with :function:`dolfin_dg.primal.aero.compressible_euler`
    to compose an interior penalty formulation of viscous compressible flow.
    """

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(u, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(u, flux):
        rho, rhou, rhoE = dolfin_dg.aero.conserved_variables(U)
        u = rhou / rho

        grad_rho = flux[0, :]
        grad_rhou = ufl.as_tensor([flux[j, :] for j in range(1, 3)])
        grad_rhoE = flux[3, :]

        # Quotient rule to find grad(u) and grad(E)
        grad_u = (grad_rhou * rho - ufl.outer(rhou, grad_rho)) / rho ** 2
        grad_E = (grad_rhoE * rho - rhoE * grad_rho) / rho ** 2

        tau = mu * (grad_u + grad_u.T - 2.0 / 3.0 * (
            ufl.tr(grad_u)) * ufl.Identity(2))
        K_grad_T = mu * gamma / Pr * (grad_E - ufl.dot(u, grad_u))

        res = ufl.as_tensor([ufl.zero(2),
                             *(tau[d, :] for d in range(2)),
                             tau * u + K_grad_T])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return -flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, U, v)
    return fos