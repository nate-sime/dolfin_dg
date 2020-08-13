import ufl

import dolfin_dg
from dolfin_dg import DGFemSIPG, homogeneity_tensor
from dolfin_dg.dg_form import DGFemStokesTerm


class NitscheBoundary:
    """Utility class for weakly enforcing boundary conditions by Nitsche's method.
    This class manages a
    :class:`dolfin_dg.dg_form.DGClassicalSecondOrderDiscretisation` to
    automatically formulate the boundary terms.
    """

    def __init__(self, F_v, u, v, C_IP=None, DGFemClass=None):
        """
        Parameters
        ----------
        F_v
            Two argument viscous flux function ``F_v(u, grad_u)``
        u
            Solution variable
        v
            Test function
        C_IP
            Interior penalty term
        DGFemClass
            Instance of
            :class:`dolfin_dg.dg_form.DGClassicalSecondOrderDiscretisation` used
            to generate the boundary formulation
        """
        if C_IP is None:
            C_IP = dolfin_dg.dg_form.generate_default_sipg_penalty_term(u)

        G = homogeneity_tensor(F_v, u)
        n = ufl.FacetNormal(u.ufl_domain())

        if DGFemClass is None:
            DGFemClass = DGFemSIPG
        vt = DGFemClass(F_v, u, v, C_IP, G, n)

        self.vt = vt

    def nitsche_bc_residual(self, u_bc, ds):
        """Weakly impose Dirichlet boundary data on a given boundary

        Parameters
        ----------
        u_bc
            UFL expression of Dirichlet boundary data
        ds
            Boundary measure

        Returns
        -------
        The corresponding boundary terms in the residual FE formulation
        """
        return self.vt.exterior_residual(u_bc, ds)

    def nitsche_bc_residual_on_interior(self, u_bc, dS):
        """Weakly impose Dirichlet boundary data on a given *interior*
        boundary.

        Parameters
        ----------
        u_bc
            UFL expression of Dirichlet boundary data
        dS
            Boundary measure

        Returns
        -------
        The corresponding boundary terms in the residual FE formulation
        """
        return self.vt.exterior_residual_on_interior(u_bc, dS)


class StokesNitscheBoundary:
    """Utility class for weakly enforcing boundary conditions by Nitsche's
    method, *specific to the Stokes system*. This class manages a
    :class:`dolfin_dg.dg_form.DGClassicalSecondOrderDiscretisation` to
    automatically formulate the boundary terms. Additional boundary terms
    which are required by the constraint of the continuity equation are also
    generated.
    """

    def __init__(self, F_v, u, p, v, q, C_IP=None, delta=-1, block_form=False):
        r"""
        Parameters
        ----------
        F_v
            Two argument viscous flux function ``F_v(u, grad_u)``
        u
            Velocity solution variable
        p
            Pressure solution variable
        v
            Velocity test function
        q
            Pressure test function
        C_IP
            Interior penalty term
        delta
            Value :math:`\delta \in \{-1, 0, 1\} corresponding to SIPG,
            NIPG and BO (Baumann-Oden) formulations, respectively.
        block_form
            If true, generated FE residual formulations will be returned in
            block form
        """

        if C_IP is None:
            C_IP = dolfin_dg.dg_form.generate_default_sipg_penalty_term(u)

        G = homogeneity_tensor(F_v, u)
        n = ufl.FacetNormal(u.ufl_domain())

        vt = DGFemStokesTerm(F_v, u, p, v, q, C_IP, G, n, delta,
                             block_form=block_form)

        self.vt = vt

    def nitsche_bc_residual(self, u_bc, ds):
        """Weakly impose Dirichlet boundary data on a given boundary

        Parameters
        ----------
        u_bc
            UFL expression of Dirichlet boundary data
        ds
            Boundary measure

        Returns
        -------
        The corresponding boundary terms in the residual FE formulation
        """
        return self.vt.exterior_residual(u_bc, ds)

    def nitsche_bc_residual_on_interior(self, u_bc, dS):
        """Weakly impose Dirichlet boundary data on a given *interior*
        boundary.

        Parameters
        ----------
        u_bc
            UFL expression of Dirichlet boundary data
        dS
            Boundary measure

        Returns
        -------
        The corresponding boundary terms in the residual FE formulation
        """
        return self.vt.exterior_residual_on_interior(u_bc, dS)

    def slip_nitsche_bc_residual(self, u_bc, f2, ds):
        r"""Weakly import Dirichlet data on the *normal* component of the
        solution, e.g., :math:`\vec{u} \cdot \vec{n} = \vec{u}_\text{bc} \cdot
        \vec{n}`

        Parameters
        ----------
        u_bc
            UFL expression of Dirichlet boundary data
        f2
            Tangential component of the stress. Can also be interpreted as
            the Neumann condition to be enforced in the tangential direction
        ds
            Boundary measure

        Returns
        -------
        The corresponding boundary terms in the residual FE formulation
        """
        return self.vt.slip_exterior_residual(u_bc, f2, ds)

    def slip_nitsche_bc_residual_on_interior(self, u_bc, f2, dS):
        r"""Weakly import Dirichlet data on the *normal* component of the
        solution, e.g., :math:`\vec{u} \cdot \vec{n} = \vec{u}_\text{bc} \cdot
        \vec{n}` on an *interior* boundary

        Parameters
        ----------
        u_bc
            UFL expression of Dirichlet boundary data
        f2
            Tangential component of the stress. Can also be interpreted as
            the Neumann condition to be enforced in the tangential direction
        dS
            Boundary measure

        Returns
        -------
        The corresponding boundary terms in the residual FE formulation
        """
        return self.vt.slip_exterior_residual_on_interior(u_bc, f2, dS)
