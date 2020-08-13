r"""
This module provides utility functions for compressible flow simulations where
the solution variable, :math:`U`, is comprised of the conserved variables

    .. math::

        \mathbf{U} =
        \begin{pmatrix}
            \rho \\
            \rho u \\
            \rho E
        \end{pmatrix}.
"""

from ufl import as_vector, dot


def conserved_variables(U):
    r"""
    Parameters
    ----------
    U
        Solution vector

    Returns
    -------
    Conserved variables :math:`(\rho, \rho u, \rho E)`
    """
    sz = len(U)
    rho, rhoE = U[0], U[sz-1]
    dim = U.ufl_shape[0] - 2
    rhou = as_vector([U[j] for j in range(1, dim + 1)])
    return rho, rhou, rhoE


def flow_variables(U):
    r"""
    Parameters
    ----------
    U
        Solution vector

    Returns
    -------
    Flow variables :math:`(\rho, u, E)`
    """
    rho, rhou, rhoE = conserved_variables(U)
    return rho, rhou/rho, rhoE/rho


def pressure(U, gamma):
    """
    Parameters
    ----------
    U
        Solution vector
    gamma
        Ratio of specific heats (typically 1.4 in air)

    Returns
    -------
    The pressure
    """
    rho, u, E = flow_variables(U)
    p = (gamma - 1.0)*rho*(E - 0.5*dot(u, u))
    return p


def enthalpy(U, gamma):
    """
    Parameters
    ----------
    U
        Solution vector
    gamma
        Ratio of specific heats (typically 1.4 in air)

    Returns
    -------
    Enthalpy
    """
    sz = len(U)
    rho, E = U[0], U[sz-1]/U[0]
    p = pressure(U, gamma)
    H = E + p/rho
    return H


def speed_of_sound(p, rho, gamma):
    """
    Parameters
    ----------
    p
        Pressure
    rho
        Density
    gamma
        Ratio of specific heats (typically 1.4 in air)

    Returns
    -------
    Speed of sound in the medium
    """
    return abs(gamma*p/rho)**0.5


def effective_reynolds_number(Re_0, M_0, gamma):
    """
    Parameters
    ----------
    Re_0
        Reference Reynolds numbers in compressible Navier-Stokes equations
    M_0
        Reference Mach number
    gamma
        Ratio of specific heats (typically 1.4 in air)

    Returns
    -------
    The effective Reynolds number
    """
    return Re_0/(gamma**0.5*M_0)


def energy_density(p, rho, u, gamma):
    """
    Parameters
    ----------
    p
        Pressure
    rho
        Density
    u
        Velocity
    gamma
        Ratio of specific heats (typically 1.4 in air)

    Returns
    -------
    Specific energy density
    """
    return p/(gamma-1.0) + 0.5*rho*dot(u, u)


def subsonic_inflow(rho_in, u_in, U, gamma):
    """
    Generate the necessary function to enforce on the boundary so to impose
    subsonic inflow

    Parameters
    ----------
    rho_in
        Prescribed inlet density
    u_in
        Prescribed inlet velocity
    U
        Solution vector
    gamma
        Ratio of specific heats (typically 1.4 in air)


    Returns
    -------
    Subsonic inflow boundary condition function
    """
    p = pressure(U, gamma)
    rhoE_in = energy_density(p, rho_in, u_in, gamma)
    return as_vector([rho_in] +
                     [rho_in*u_in[j] for j in range(u_in.ufl_shape[0])] +
                     [rhoE_in])


def subsonic_outflow(p_out, U, gamma):
    """
    Generate the necessary function to enforce on the boundary so to impose
    subsonic outflow

    Parameters
    ----------
    p_out
        Prescribed outlet pressure
    U
        Solution vector
    gamma
        Ratio of specific heats (typically 1.4 in air)

    Returns
    -------
    Subsonic outflow boundary condition function
    """
    rho, u, E = flow_variables(U)
    rhoE_out = energy_density(p_out, rho, u, gamma)
    dim = U.ufl_shape[0] - 2
    return as_vector([rho] + [U[j] for j in range(1, dim+1)] + [rhoE_out])


def no_slip(U):
    """
    Generate the necessary function to enforce no slip boundary conditions

    Parameters
    ----------
    U
        Solution vector

    Returns
    -------
    No slip boundary condition function
    """
    rho, rhou, rhoE = conserved_variables(U)
    dim = U.ufl_shape[0] - 2
    return as_vector([rho] + [0]*dim + [rhoE])
