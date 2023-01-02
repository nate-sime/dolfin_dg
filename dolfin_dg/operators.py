import inspect

import ufl
from ufl import (
    grad, inner, curl, dot, as_vector, tr, Identity, variable, diff, exp,
    Measure, FacetNormal
)

from dolfin_dg import aero, generate_default_sipg_penalty_term, \
    homogeneity_tensor
from dolfin_dg.dg_form import (
    DGFemTerm, DGFemCurlTerm, DGFemSIPG, DGFemStokesTerm
)
from dolfin_dg.fluxes import LocalLaxFriedrichs


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


class DGFemFormulation:
    """Abstract base class for automatic formulation of a DG FEM
    formulation
    """

    @staticmethod
    def mesh_dimension(mesh):
        # the DUNE way
        if hasattr(mesh, "dimension"):
            return mesh.dimension
        else:
        # the Fenics/Dolfin way
            try:
                return mesh.geometry().dim()
            except AttributeError:
                return mesh.geometric_dimension()

    def __init__(self, mesh, fspace, bcs):
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
        """
        if not hasattr(bcs, '__len__'):
            bcs = [bcs]
        self.mesh = mesh
        self.fspace = fspace
        self.dirichlet_bcs = [bc for bc in bcs if isinstance(bc, DGDirichletBC)]
        self.neumann_bcs = [bc for bc in bcs if isinstance(bc, DGNeumannBC)]

    def ufl_domain(self):
        try:
            # the Fenics/Dolfin way
            return self.mesh.ufl_domain()
        except AttributeError:
            # the DUNE way (the FE space is also ufl domain)
            return self.fspace

    def generate_fem_formulation(self, u, v, dx=None, dS=None, vt=None):
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
        vt
            A specific implementation of
            :class:`dolfin_dg.dg_form.DGClassicalSecondOrderDiscretisation`

        Returns
        -------
        The UFL representation of the DG FEM formulation
        """
        raise NotImplementedError('Function not yet implemented')


class EllipticOperator(DGFemFormulation):
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
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_v = F_v

    def generate_fem_formulation(self, u, v, dx=None, dS=None, vt=None,
                                 penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.ufl_domain())
        G = homogeneity_tensor(self.F_v, u)

        if penalty is None:
            penalty = generate_default_sipg_penalty_term(u)

        if vt is None:
            vt = DGFemSIPG(self.F_v, u, v, penalty, G, n)

        if inspect.isclass(vt):
            vt = vt(self.F_v, u, v, penalty, G, n)

        assert(isinstance(vt, DGFemTerm))

        residual = inner(self.F_v(u, grad(u)), grad(v))*dx
        residual += vt.interior_residual(dS)

        for dbc in self.dirichlet_bcs:
            residual += vt.exterior_residual(
                dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += vt.neumann_residual(
                dbc.get_function(), dbc.get_boundary())

        return residual


class PoissonOperator(EllipticOperator):
    r"""Specific implementation of
    :class:`dolfin_dg.operators.EllipticOperator` for the Poisson operator:

    .. math :: - \nabla \cdot \kappa \nabla u
    """

    def __init__(self, mesh, fspace, bcs, kappa=1):
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
        kappa
            (Potentially nonlinear) diffusion coefficient
        """
        def F_v(u, grad_u):
            return kappa*grad_u

        EllipticOperator.__init__(self, mesh, fspace, bcs, F_v)


class MaxwellOperator(DGFemFormulation):
    r"""Base class for the automatic generation of a DG formulation for
    the underlying elliptic (2nd order) operator of the form

    .. math:: \nabla \times \mathcal{F}^m(u, \nabla \times u)
    """

    def __init__(self, mesh, fspace, bcs, F_m):
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
        F_m
            Two argument function ``F_m(u, curl_u)`` corresponding to the
            viscous flux term
        """
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_m = F_m

    def generate_fem_formulation(self, u, v, dx=None, dS=None, penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.ufl_domain())
        curl_u = variable(curl(u))
        G = diff(self.F_m(u, curl_u), curl_u)
        penalty = generate_default_sipg_penalty_term(u)

        ct = DGFemCurlTerm(self.F_m, u, v, penalty, G, n)

        residual = inner(self.F_m(u, curl(u)), curl(v))*dx
        residual += ct.interior_residual(dS)

        for dbc in self.dirichlet_bcs:
            residual += ct.exterior_residual(
                dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += ct.neumann_residual(
                dbc.get_function(), dbc.get_boundary())

        return residual


class HyperbolicOperator(DGFemFormulation):
    r"""Base class for the automatic generation of a DG formulation for
    the underlying hyperbolic (1st order) operator of the form

    .. math:: \nabla \cdot \mathcal{F}^c(u)
    """

    def __init__(self, mesh, V, bcs, F_c=lambda u: u,
                 H=LocalLaxFriedrichs(lambda u, n: inner(u, n))):
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
        DGFemFormulation.__init__(self, mesh, V, bcs)
        self.F_c = F_c
        self.H = H

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
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

        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.ufl_domain())

        F_c_eval = self.F_c(u)
        if len(F_c_eval.ufl_shape) == 0:
            F_c_eval = as_vector((F_c_eval,))
        residual = -inner(F_c_eval, grad(v))*dx

        self.H.setup(self.F_c, u('+'), u('-'), n('+'))
        residual += inner(self.H.interior(self.F_c, u('+'), u('-'), n('+')),
                          (v('+') - v('-')))*dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, gD, n)
            residual += inner(self.H.exterior(self.F_c, u, gD, n), v)*dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v)*dSN

        return residual


class SpacetimeBurgersOperator(HyperbolicOperator):
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

        def F_c(u):
            return as_vector((u**2/2, u))

        if flux is None:
            flux = LocalLaxFriedrichs(lambda u, n: u*n[0] + n[1])

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, flux)


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

        dim = self.mesh_dimension( mesh )

        def F_c(U):
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            H = aero.enthalpy(U, gamma=gamma)

            inertia = rho*ufl.outer(u, u) + p*Identity(dim)
            res = ufl.as_tensor([rho*u,
                                 *[inertia[d, :] for d in range(dim)],
                                 rho*H*u])
            return res

        def alpha(U, n):
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            c = aero.speed_of_sound(p, rho, gamma=gamma)
            lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c,
                                    LocalLaxFriedrichs(alpha))


class CompressibleNavierStokesOperator(EllipticOperator,
                                       CompressibleEulerOperator):
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
        dim = self.mesh_dimension( mesh )

        if not hasattr(bcs, '__len__'):
            bcs = [bcs]
        self.adiabatic_wall_bcs = [bc for bc in bcs
                                   if isinstance(bc, DGAdiabticWallBC)]

        def F_v(U, grad_U):
            rho, rhou, rhoE = aero.conserved_variables(U)
            u = rhou/rho

            grad_rho = grad_U[0, :]
            grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim + 1)])
            grad_rhoE = grad_U[-1, :]

            # Quotient rule to find grad(u) and grad(E)
            grad_u = (grad_rhou*rho - ufl.outer(rhou, grad_rho))/rho**2
            grad_E = (grad_rhoE*rho - rhoE*grad_rho)/rho**2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(dim))
            K_grad_T = mu*gamma/Pr*(grad_E - dot(u, grad_u))

            res = ufl.as_tensor([ufl.zero(dim),
                                 *(tau[d, :] for d in range(dim)),
                                 tau * u + K_grad_T])
            return res

        # Specialised adiabatic wall BC
        def F_v_adiabatic(U, grad_U):
            rho, rhou, rhoE = aero.conserved_variables(U)
            u = rhou/rho

            grad_rho = grad_U[0, :]
            grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim + 1)])
            grad_u = (grad_rhou * rho - ufl.outer(rhou, grad_rho)) / rho ** 2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(dim))

            res = ufl.as_tensor([ufl.zero(dim),
                                 *(tau[d, :] for d in range(dim)),
                                 tau * u])
            return res

        self.F_v_adiabatic = F_v_adiabatic

        CompressibleEulerOperator.__init__(self, mesh, V, bcs, gamma)
        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None, penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        residual = EllipticOperator.generate_fem_formulation(
            self, u, v, dx=dx, dS=dS, penalty=penalty)
        residual += CompressibleEulerOperator.generate_fem_formulation(
            self, u, v, dx=dx, dS=dS)

        # Specialised adiabatic wall boundary condition
        for bc in self.adiabatic_wall_bcs:
            n = FacetNormal(self.mesh)

            u_gamma = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, u_gamma, n)
            residual += inner(self.H.exterior(self.F_c, u, u_gamma, n), v)*dSD

            if penalty is None:
                penalty = generate_default_sipg_penalty_term(u)
            G_adiabitic = homogeneity_tensor(self.F_v_adiabatic, u)
            vt_adiabatic = DGFemSIPG(
                self.F_v_adiabatic, u, v, penalty, G_adiabitic, n)

            residual += vt_adiabatic.exterior_residual(u_gamma, dSD)

        return residual


def V_to_U(V, gamma):
    """Map the entropy variable formulation to the mass, momentum, energy
    variables.

    Parameters
    ----------
    V
        Entropy variables
    gamma
        Ratio of specific heats

    Returns
    -------
    mass, momentum and energy variables
    """
    V1, V2, V3, V4 = V
    U = as_vector([-V4, V2, V3, 1 - 0.5*(V2**2 + V3**2)/V4])
    s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
    rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*exp(-s/(gamma-1))
    U = U*rhoi
    return U


class CompressibleEulerOperatorEntropyFormulation(HyperbolicOperator):
    r"""Specific implementation of
    :class:`dolfin_dg.operators.HyperbolicOperator` for the entropy variable
    formulation of the compressible Euler operator
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

        dim = self.mesh_dimension( mesh )

        def F_c(V):
            V = variable(V)
            U = V_to_U(V, gamma)
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            H = aero.enthalpy(U, gamma=gamma)

            inertia = rho*ufl.outer(u, u) + p*Identity(dim)
            res = ufl.as_tensor([rho*u,
                                 *[inertia[d, :] for d in range(dim)],
                                 rho*H*u])
            return res

        def alpha(V, n):
            U = V_to_U(V, gamma)
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            c = aero.speed_of_sound(p, rho, gamma=gamma)
            lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c,
                                    LocalLaxFriedrichs(alpha))


class CompressibleNavierStokesOperatorEntropyFormulation(
        EllipticOperator,
        CompressibleEulerOperatorEntropyFormulation):
    r"""Specific implementation of
    :class:`dolfin_dg.operators.CompressibleEulerOperatorEntropyFormulation`
    and :class:`dolfin_dg.operators.EllipticOperator` for the entropy variable
    formulation of the compressible Navier-Stokes operator
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

        dim = self.mesh_dimension( mesh )

        def F_v(V, grad_V):
            V = variable(V)
            U = V_to_U(V, gamma)
            dudv = diff(U, V)
            grad_U = dot(dudv, grad_V)

            rho, rhou, rhoE = aero.conserved_variables(U)
            rho, u, E = aero.flow_variables(U)

            grad_rho = grad_U[0, :]
            grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim + 1)])
            grad_rhoE = grad_U[-1, :]

            grad_u = (grad_rhou*rho - ufl.outer(rhou, grad_rho))/rho**2
            grad_E = (grad_rhoE*rho - rhoE*grad_rho)/rho**2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(dim))
            K_grad_T = mu*gamma/Pr*(grad_E - dot(u, grad_u))

            res = ufl.as_tensor([ufl.zero(dim),
                                 *(tau[d, :] for d in range(dim)),
                                 tau * u + K_grad_T])
            return res

        CompressibleEulerOperatorEntropyFormulation.__init__(
            self, mesh, V, bcs, gamma)
        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None, penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        residual = EllipticOperator \
            .generate_fem_formulation(self, u, v, dx=dx, dS=dS, penalty=penalty)
        residual += CompressibleEulerOperatorEntropyFormulation \
            .generate_fem_formulation(self, u, v, dx=dx, dS=dS)

        return residual


class StokesOperator(DGFemFormulation):
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
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_v = F_v

    def generate_fem_formulation(self, u, v, p, q, dx=None, dS=None,
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

        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.ufl_domain())
        G = homogeneity_tensor(self.F_v, u)
        delta = -1

        if penalty is None:
            penalty = generate_default_sipg_penalty_term(u)

        vt = DGFemStokesTerm(self.F_v, u, p, v, q, penalty, G, n, delta,
                             block_form=block_form)

        residual = [ufl.inner(self.F_v(u, grad(u)), grad(v))*dx,
                    q*ufl.div(u)*dx]
        if not block_form:
            residual = sum(residual)

        def _add_to_residual(residual, r):
            if block_form:
                for j in range(len(r)):
                    residual[j] += r[j]
            else:
                residual += r
            return residual

        residual = _add_to_residual(residual, vt.interior_residual(dS))

        for dbc in self.dirichlet_bcs:
            residual = _add_to_residual(
                residual, vt.exterior_residual(dbc.get_function(),
                                               dbc.get_boundary()))

        for dbc in self.neumann_bcs:
            elliptic_neumann_term = vt.neumann_residual(
                dbc.get_function(), dbc.get_boundary())
            if block_form:
                elliptic_neumann_term = [elliptic_neumann_term, 0]
            residual = _add_to_residual(residual, elliptic_neumann_term)

        return residual
