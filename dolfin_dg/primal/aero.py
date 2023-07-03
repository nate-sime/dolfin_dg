import ufl

import dolfin_dg
import dolfin_dg.primal


# -- Standard formulation

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
    dim = U.ufl_shape[0] - 2

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_1(_, flux):
        rho, u, E = dolfin_dg.aero.flow_variables(flux)
        p = dolfin_dg.aero.pressure(flux, gamma=gamma)
        H = dolfin_dg.aero.enthalpy(flux, gamma=gamma)

        inertia = rho * ufl.outer(u, u) + p * ufl.Identity(dim)
        res = ufl.as_tensor([rho * u,
                             *[inertia[d, :] for d in range(dim)],
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
    dim = U.ufl_shape[0] - 2

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(_, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(U, flux):
        rho, rhou, rhoE = dolfin_dg.aero.conserved_variables(U)
        u = rhou / rho

        grad_rho = flux[0, :]
        grad_rhou = ufl.as_tensor([flux[j, :] for j in range(1, dim+1)])
        grad_rhoE = flux[dim+1, :]

        # Quotient rule to find grad(u) and grad(E)
        grad_u = (grad_rhou * rho - ufl.outer(rhou, grad_rho)) / rho ** 2
        grad_E = (grad_rhoE * rho - rhoE * grad_rho) / rho ** 2

        tau = mu * (grad_u + grad_u.T - 2.0 / 3.0 * (
            ufl.tr(grad_u)) * ufl.Identity(dim))
        K_grad_T = mu * gamma / Pr * (grad_E - ufl.dot(u, grad_u))

        res = ufl.as_tensor([ufl.zero(2),
                             *(tau[d, :] for d in range(dim)),
                             tau * u + K_grad_T])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(_, flux):
        return -flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, U, v)
    return fos


def compressible_navier_stokes_adiabatic_wall(U, v, gamma=1.4, mu=1, Pr=0.72):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    compressible Navier Stokes operator with adiabatic wall
     (:math:`k \nabla T = 0`):

    .. math ::
        -\nabla \cdot
        \begin{pmatrix}
        \vec{0} \\
        \sigma \\
        \sigma \vec{u}
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
    dim = U.ufl_shape[0] - 2

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(_, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(U, flux):
        rho, rhou, rhoE = dolfin_dg.aero.conserved_variables(U)
        u = rhou / rho

        grad_rho = flux[0, :]
        grad_rhou = ufl.as_tensor([flux[j, :] for j in range(1, dim+1)])
        grad_u = (grad_rhou * rho - ufl.outer(rhou, grad_rho)) / rho ** 2

        tau = mu * (grad_u + grad_u.T - 2.0 / 3.0 * (
            ufl.tr(grad_u)) * ufl.Identity(dim))

        res = ufl.as_tensor([ufl.zero(dim),
                             *(tau[d, :] for d in range(dim)),
                             tau * u])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(_, flux):
        return -flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, U, v)
    return fos


# -- Entropy formulation

def U_to_V(U, gamma):
    """Map the mass, momentum and energy variables to the entropy variables.

    Parameters
    ----------
    U
        Mass, momentum and energy variable vector
    gamma
        Ratio of specific heats

    Returns
    -------
    Entropy variable vector
    """
    rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
    i = E - 0.5*(u1**2 + u2**2)
    U1, U2, U3, U4 = U
    s = ufl.ln((gamma-1)*rho*i/(U1**gamma))
    V1 = 1/(rho*i)*(-U4 + rho*i*(gamma + 1 - s))
    V2 = 1/(rho*i)*U2
    V3 = 1/(rho*i)*U3
    V4 = 1/(rho*i)*(-U1)
    return ufl.as_vector([V1, V2, V3, V4])


def V_to_U(V, gamma):
    """Map the entropy variable formulation to the mass, momentum and energy
    variables.

    Parameters
    ----------
    V
        Entropy variables
    gamma
        Ratio of specific heats

    Returns
    -------
    Mass, momentum and energy variable vector
    """
    V1, V2, V3, V4 = V
    U = ufl.as_vector([-V4, V2, V3, 1 - 0.5*(V2**2 + V3**2)/V4])
    s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
    rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*ufl.exp(-s/(gamma-1))
    U = U*rhoi
    return U


def compressible_euler_entropy(V, v, gamma=1.4):
    r"""Generate a :class:`dolfin_dg.primal.FirstOrderSystem` for the
    compressible Euler operator entropy formulation.
    """
    dim = V.ufl_shape[0] - 2

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_1(_, flux):
        V = ufl.variable(flux)
        U = V_to_U(V, gamma)
        rho, u, E = dolfin_dg.aero.flow_variables(U)
        p = dolfin_dg.aero.pressure(U, gamma=gamma)
        H = dolfin_dg.aero.enthalpy(U, gamma=gamma)

        inertia = rho * ufl.outer(u, u) + p * ufl.Identity(dim)
        res = ufl.as_tensor([rho * u,
                             *[inertia[d, :] for d in range(dim)],
                             rho * H * u])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(u, flux):
        return flux

    F_vec = [F_0, F_1]
    L_vec = [ufl.div]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, V, v)
    return fos


def compressible_navier_stokes_entropy(V, v, gamma=1.4, mu=1, Pr=0.72):
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
    dim = V.ufl_shape[0] - 2

    @dolfin_dg.primal.first_order_flux(lambda x: x)
    def F_2(_, flux):
        return flux

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
    def F_1(V, grad_V):
        V = ufl.variable(V)
        U = V_to_U(V, gamma)
        grad_U = ufl.dot(ufl.diff(U, V), grad_V)

        rho, rhou, rhoE = dolfin_dg.aero.conserved_variables(U)
        u = rhou / rho

        grad_rho = grad_U[0, :]
        grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim+1)])
        grad_rhoE = grad_U[dim+1, :]

        # Quotient rule to find grad(u) and grad(E)
        grad_u = (grad_rhou * rho - ufl.outer(rhou, grad_rho)) / rho ** 2
        grad_E = (grad_rhoE * rho - rhoE * grad_rho) / rho ** 2

        tau = mu * (grad_u + grad_u.T - 2.0 / 3.0 * (
            ufl.tr(grad_u)) * ufl.Identity(dim))
        K_grad_T = mu * gamma / Pr * (grad_E - ufl.dot(u, grad_u))

        res = ufl.as_tensor([ufl.zero(2),
                             *(tau[d, :] for d in range(dim)),
                             tau * u + K_grad_T])
        return res

    @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
    def F_0(_, flux):
        return -flux

    F_vec = [F_0, F_1, F_2]
    L_vec = [ufl.div, ufl.grad]

    fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, V, v)
    return fos