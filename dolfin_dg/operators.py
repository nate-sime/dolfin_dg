import inspect

import ufl
from ufl import (
    grad, inner, curl, dot, as_vector, tr, Identity, variable, diff, exp,
    Measure, FacetNormal
)

from dolfin_dg import aero, generate_default_sipg_penalty_term, \
    homogeneity_tensor
from dolfin_dg.dg_form import (
    DGFemTerm, DGFemSIPG
)
import dolfin_dg.penalty
import dolfin_dg.primal


class DGBC:
    """Utility class indicating the nature of a weakly imposed boundary
    condition. The actual implementation is application dependent.
    """

    def __init__(self, boundary, function):
        """
        Parameters
        ----------
        boundary
            The UFL measure
        function
            The function or expression to be weakly imposed
        """
        self.__boundary = boundary
        self.__function = function

    def get_boundary(self):
        return self.__boundary

    def get_function(self):
        return self.__function

    def __repr__(self):
        return '%s(%s: %s)' % (self.__class__.__name__,
                               self.get_boundary(),
                               self.get_function())


class DGDirichletBC(DGBC):
    """Indicates weak imposition of Dirichlet BCs
    """
    pass


class DGNeumannBC(DGBC):
    """Indicates weak imposition of Neumann or outflow BCs
    """
    pass


class DGDirichletNormalBC(DGBC):
    """Indicates weak imposition of Dirichlet BCs to be imposed on the normal
    component of the solution. E.g. free slip BCs
    """
    pass


class DGAdiabticWallBC(DGBC):
    """Indicates weak imposition of specialised Dirichlet BC case in
    compressible flow where there is zero energy flux through the boundary
    """
    pass


def mesh_dimension(mesh):
    if hasattr(mesh, "dimension"):
        # the DUNE way
        return mesh.dimension
    elif hasattr(mesh, "ufl_domain"):
        # the dolfin/dolfinx way
        return mesh.ufl_domain().geometric_dimension()
    elif hasattr(mesh, "geometric_dimension"):
        # the UFL way
        return mesh.geometric_dimension()
    else:
        # the firedrake & legacy fenics/dolfin way
        try:
            return mesh.geometry().dim()
        except AttributeError:
            return mesh.geometric_dimension()


class EllipticOperator:
    r"""Base class for the automatic generation of a DG formulation for
    the underlying elliptic (2nd order) operator of the form

    .. math:: -\nabla \cdot \mathcal{F}^v(u, \nabla u)
    """

    def __init__(self, mesh, fspace, bcs, F_v):
        """
        Parameters
        ----------
        mesh
            Problem mesh
        fspace
            Problem function space in which the solution is formulated and
            sought
        bcs
            List of :class:`dolfin_dg.operators.DGBC` to be weakly imposed and
            included in the formulation
        F_v
            Two argument function ``F_v(u, grad_u)`` corresponding to the
            viscous flux term
        """
        self.bcs = bcs
        self.F_v = F_v

    def generate_fem_formulation(
            self, u, v, dx=ufl.dx, dS=ufl.dS, vt=None, c_ip=20.0,
            h_measure=None):
        @dolfin_dg.primal.first_order_flux(lambda x: x)
        def F_2(_, flux):
            return flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(self.F_v(x)))
        def F_0(_, flux):
            return -flux

        wrapped_F_v = dolfin_dg.primal.first_order_flux(
            lambda x: ufl.grad(F_2(x)))(self.F_v)
        F_vec = [F_0, wrapped_F_v, F_2]
        L_vec = [ufl.div, ufl.grad]

        fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)

        F = fos.domain(dx)
        F += apply_interior_penalty(
            fos, self.bcs, c_ip=c_ip, h_measure=h_measure, dS=dS)
        return F


def apply_interior_penalty(fos, bcs, c_ip=20.0, h_measure=None, dS=ufl.dS):
    if not isinstance(bcs, (tuple, list)):
        bcs = [bcs]

    u = fos.u
    alpha, _ = dolfin_dg.penalty.interior_penalty(
        fos, u, u, c_ip=c_ip, h_measure=h_measure)
    F = fos.interior([alpha], dS=dS)

    for bc in bcs:
        bdry, function = bc.get_boundary(), bc.get_function()
        if isinstance(bc, DGDirichletBC):
            _, alpha_ext = dolfin_dg.penalty.interior_penalty(
                fos, u, function, c_ip=c_ip, h_measure=h_measure)
            F += fos.exterior([alpha_ext], function, ds=bdry)
    return F


class PoissonOperator:
    r"""
    .. math:: \nabla \cdot \kappa \nabla (\cdot)
    """

    def __init__(self, mesh, fspace, bcs, kappa=1):
        self.kappa = kappa
        self.bcs = bcs

    def generate_fem_formulation(self, u, v, dx=ufl.dx, dS=ufl.dS, vt=None,
                                 c_ip=20.0, h_measure=None):
        import dolfin_dg.primal.simple
        fos = dolfin_dg.primal.simple.diffusion(u, v, self.kappa)
        F = fos.domain(dx)
        F += apply_interior_penalty(
            fos, self.bcs, c_ip=c_ip, h_measure=h_measure, dS=dS)
        return F


class MaxwellOperator:
    r"""
    .. math:: \nabla \times \nabla \times (\cdot)
    """

    def __init__(self, mesh, fspace, bcs, F_m):
        self.bcs = bcs
        self.F_m = F_m

    def generate_fem_formulation(
            self, u, v, dx=ufl.dx, dS=ufl.dS, c_ip=20.0, h_measure=None):
        import dolfin_dg.primal.simple
        fos = dolfin_dg.primal.simple.maxwell(u, v)
        F = fos.domain(dx)
        F += apply_interior_penalty(
            fos, self.bcs, c_ip=c_ip, h_measure=h_measure, dS=dS)
        return F


def apply_facet_flux(fos, bcs, lambdas, dS=ufl.dS):
    u = fos.u

    if not isinstance(bcs, (list, tuple)):
        bcs = [bcs]

    alpha, _ = dolfin_dg.penalty.local_lax_friedrichs_penalty(
        lambdas, u, u)
    F = fos.interior([-alpha], dS=dS)

    for bc in bcs:
        bdry, function = bc.get_boundary(), bc.get_function()
        if isinstance(bc, (DGDirichletBC, DGAdiabticWallBC)):
            _, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
                lambdas, u, function)
            F += fos.exterior([-alpha_ext], function, ds=bdry)
        elif isinstance(bc, DGNeumannBC):
            _, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
                lambdas, u, u)
            F += fos.exterior([-alpha_ext], function, ds=bdry)
    return F


class HyperbolicOperator:
    r"""Base class for the automatic generation of a DG formulation for
    the underlying hyperbolic (1st order) operator of the form

    .. math:: \nabla \cdot \mathcal{F}^c(u)
    """

    def __init__(self, mesh, V, bcs, F_c=lambda u: u, lambdas=None):
        """
        Parameters
        ----------
        mesh
            Problem mesh
        fspace
            Problem function space in which the solution is formulated and
            sought
        bcs
            List of :class:`dolfin_dg.operators.DGBC` to be weakly imposed and
            included in the formulation
        F_c
            One argument function ``F_c(u)`` corresponding to the
            convective flux term
        H
            An instance of a :class:`dolfin_dg.fluxes.ConvectiveFlux`
            describing the convective flux scheme to employ
        """
        self.bcs = bcs
        self.F_c = dolfin_dg.primal.first_order_flux(lambda x: x)(F_c)
        self.lambdas = lambdas

    def generate_fem_formulation(self, u, v, dx=ufl.dx, dS=ufl.dS):
        """Automatically generate the DG FEM formulation

        Parameters
        ----------
        u
            Solution variable
        v
            Test function
        dx
            Volume integration measure
        dS
            Interior facet integration measure

        Returns
        -------
        The UFL representation of the DG FEM formulation
        """
        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(self.F_c(x)))
        def F_0(u, flux):
            return flux

        F_vec = [F_0, self.F_c]
        L_vec = [ufl.div]

        fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)

        F = fos.domain(dx)

        n = ufl.FacetNormal(u.ufl_domain())
        if self.lambdas is None:
            lambdas = ufl.dot(ufl.diff(fos.F_vec[1](u), u), n)
        else:
            lambdas = self.lambdas
        F += apply_facet_flux(fos, self.bcs, lambdas)
        return F


class SpacetimeBurgersOperator:
    r"""Specific implementation of
    :class:`dolfin_dg.operators.HyperbolicOperator` for the spacetime Burgers
    operator where :math:`t=y`

    .. math ::
        \nabla \cdot
        \begin{pmatrix}
        \frac{1}{2} u^2 \\
        u
        \end{pmatrix}
    """

    def __init__(self, mesh, V, bcs, flux=None):
        self.bcs = bcs

    def generate_fem_formulation(
            self, u, v, dx=ufl.dx, dS=ufl.dS):
        import dolfin_dg.primal

        @dolfin_dg.primal.first_order_flux(lambda x: x)
        def F_1(_, flux):
            return as_vector((flux**2/2, flux))

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
        def F_0(u, flux):
            return flux

        F_vec = [F_0, F_1]
        L_vec = [ufl.div]

        fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)

        F = fos.domain(dx)

        n = ufl.FacetNormal(u.ufl_domain())
        lambdas = ufl.dot(ufl.diff(fos.F_vec[1](u), u), n)
        F += apply_facet_flux(fos, self.bcs, lambdas, dS=dS)
        return F


class CompressibleEulerOperator(HyperbolicOperator):
    r"""Specific implementation of
    :class:`dolfin_dg.operators.HyperbolicOperator` for the compressible Euler
    operator

    .. math ::
        \nabla \cdot
        \begin{pmatrix}
        \rho \vec{u} \\
        \rho \vec{u} \otimes \vec{u} + p I \\
        (\rho E + p) \vec{u}
        \end{pmatrix}
    """

    def __init__(self, mesh, V, bcs, gamma=1.4):
        """
        Parameters
        ----------
        mesh
            Problem mesh
        V
            Problem function space in which the solution is formulated and
            sought
        bcs
            List of :class:`dolfin_dg.operators.DGDC` to be weakly imposed
            and included in the formulation
        gamma
            Ratio of specific heats
        """
        self.gamma = gamma
        self.bcs = bcs

    def generate_fem_formulation(self, U, v, dx=ufl.dx, dS=ufl.dS):
        n = ufl.FacetNormal(U.ufl_domain())
        gamma = self.gamma

        rho, u, E = aero.flow_variables(U)
        p = aero.pressure(U, gamma=gamma)
        c = aero.speed_of_sound(p, rho, gamma=gamma)
        lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]

        import dolfin_dg.primal.aero
        fos = dolfin_dg.primal.aero.compressible_euler(U, v, gamma=gamma)

        F = fos.domain(dx)
        F += apply_facet_flux(fos, self.bcs, lambdas)
        return F



class CompressibleNavierStokesOperator:
    r"""Specific implementation of
    :class:`dolfin_dg.operators.EllipticOperator` and
    :class:`dolfin_dg.operators.CompressibleEulerOperator` for the compressible
    Navier-Stokes operator

    .. math ::
        \nabla \cdot \left(
        \begin{pmatrix}
        \rho \vec{u} \\
        \rho \vec{u} \otimes \vec{u} + p I \\
        (\rho E + p) \vec{u}
        \end{pmatrix}
        -
        \begin{pmatrix}
        \vec{0} \\
        \sigma \\
        \sigma \vec{u} + \kappa \nabla T
        \end{pmatrix}
        \right)

    where :math:`\sigma = \mu \left( \nabla \vec{u} + \nabla \vec{u}^\top
    - \frac{2}{3} (\nabla \cdot \vec{u}) I \right)`.
    """

    def __init__(self, mesh, V, bcs, gamma=1.4, mu=1.0, Pr=0.72):
        """
        Parameters
        ----------
        mesh
            Problem mesh
        V
            Problem function space in which the solution is formulated and
            sought
        bcs
            List of :class:`dolfin_dg.operators.DGDC` to be weakly imposed
            and included in the formulation
        gamma
            Ratio of specific heats
        mu
            Viscosity
        Pr
            Prandtl number
        """
        # dim = mesh_dimension(mesh)
        self.bcs = bcs
        self.gamma = gamma
        self.mu = mu
        self.Pr = Pr

    def generate_fem_formulation(self, U, v, dx=ufl.dx, dS=ufl.dS, vt=None,
                                 c_ip=20.0, h_measure=None):
        # -- Euler component
        ce = CompressibleEulerOperator(None, None, self.bcs, gamma=self.gamma)
        F = ce.generate_fem_formulation(U, v, dx=dx, dS=dS)

        import dolfin_dg.primal.aero
        fos = dolfin_dg.primal.aero.compressible_navier_stokes(
            U, v, gamma=self.gamma, mu=self.mu, Pr=self.Pr)
        F += fos.domain(dx)
        F += apply_interior_penalty(
            fos, self.bcs, c_ip=c_ip, h_measure=h_measure, dS=dS)

        fos_adiabatic = \
            dolfin_dg.primal.aero.compressible_navier_stokes_adiabatic_wall(
                U, v, gamma=self.gamma, mu=self.mu, Pr=self.Pr)
        for bc in self.bcs:
            bdry, function = bc.get_boundary(), bc.get_function()
            if isinstance(bc, DGAdiabticWallBC):
                _, alpha_ext = dolfin_dg.penalty.interior_penalty(
                    fos_adiabatic, U, function, c_ip=c_ip, h_measure=h_measure)
                F += fos_adiabatic.exterior([alpha_ext], function, ds=bdry)

        return F


class StokesOperator:
    r"""Base class for the Stokes operator

    .. math::

        \nabla \cdot
        \begin{pmatrix}
        - \mathcal{F}^v(\vec{u}, \nabla \vec{u})
        \vec{u}
        \end{pmatrix}
        =
        \begin{pmatrix}
        \vec{f} \\
        0
        \end{pmatrix}

    which is discretised into the saddle point system

    .. math::

        (\mathcal{F}^v(\vec{u}, \nabla \vec{u}), \vec{v})
        &= (\vec{f}, \vec{v}) \\
        (\nabla \cdot \vec{u}, q) &= 0

    """

    def __init__(self, mesh, fspace, bcs, F_v):
        self.bcs = bcs

    def generate_fem_formulation(self, u, v, p, q, eta, dx=ufl.dx, dS=ufl.dS,
                                 penalty=None, block_form=False):
        """Automatically generate the DG FEM formulation

        Parameters
        ----------
        u
            Solution velocity variable
        v
            Velocity test function
        p
            Solution pressure variable
        q
            Pressure test function
        dx
            Volume integration measure
        dS
            Interior facet integration measure
        penalty
            Interior penalty parameter
        block_form
            If true a list of formulations is returned, each element of the list
            corresponds to a block of the Stokes residual

        Returns
        -------
        The UFL representation of the DG FEM formulation
        """
        import dolfin_dg.primal.stokes

        fos_mom = dolfin_dg.primal.stokes.stokes_stress(u, v, p, eta(u))
        fos_mass = dolfin_dg.primal.stokes.stokes_continuity(u, q)

        if penalty is None:
            penalty = generate_default_sipg_penalty_term(u)

        F = fos_mom.domain(dx=dx)
        F += fos_mom.interior([penalty("+")], dS=dS)

        F += fos_mass.domain()
        F += fos_mass.interior([])

        for bc in self.bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                F += fos_mom.exterior(
                    [penalty], bc.get_function(), ds=bc.get_boundary())
                F += fos_mass.exterior(
                    [], bc.get_function(), ds=bc.get_boundary())
            elif isinstance(bc, dolfin_dg.DGNeumannBC):
                F += - ufl.inner(bc.get_function(), v) * bc.get_boundary()

        return F
