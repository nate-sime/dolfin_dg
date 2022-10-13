import abc
import inspect

import dolfin_dg
import ufl
from ufl import (
    as_matrix, outer, inner, replace, grad, variable, diff, dot, curl, div)

from dolfin_dg.dg_ufl import (
    apply_dg_operators, avg, tensor_jump, jump, tangent_jump, dg_cross)


def normal_proj(u, n):
    r"""
    Parameters
    ----------
    u
        Vector expression
    n
        Normal vector

    Returns
    -------
    Normal projection of the vector expression :math:`(n \otimes n) u`
    """
    return ufl.outer(n, n) * u


def tangential_proj(u, n):
    r"""

    Parameters
    ----------
    u
        Vector expression
    n
        Normal vector

    Returns
    -------
    Tangential projection of the vector expression :math:`(I - n \otimes n) u`
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def hyper_tensor_product(G, tau):
    r"""
    Computes the product

    .. math::

        (G \tau)_{ik} = \sum_{j=1}^m \sum_{l=1}^d (G_{kl})_{ij} \tau_{jl}

    where :math:`m` and :math:`d` are the number of rows and columns in
    :math:`\tau`, respectively. Typically :math:`d` is the spatial dimension
    and :math:`m` is the dimension of the solution vector when used in the
    automatic generation of DG FE formulations by homogenisation.

    Parameters
    ----------
    G
        Homogeneity tensor
    tau
        Tensor to be multiplied

    Returns
    -------
    G * tau
    """
    if len(G.ufl_shape) == 0:
        if not len(tau.ufl_shape) == 0:
            raise IndexError("G is scalar, tau has shape: %s"
                             + str(tau.ufl_shape))
        return G*tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return dot(G, tau.T).T
    elif ufl.rank(tau) == 1:
        return dot(G, tau)
    m, d = tau.ufl_shape
    return as_matrix([[inner(G[i, k, :, :], tau) for k in range(d)]
                      for i in range(m)])


def hyper_tensor_T_product(G, tau):
    r"""
    Computes the transpose product

    .. math::

        (G^\top \tau)_{jl} = \sum_{i=1}^m \sum_{k=1}^d (G_{kl})_{ij} \tau_{ik}

    where :math:`m` and :math:`d` are the number of rows and columns in
    :math:`\tau`, respectively. Typically :math:`d` is the spatial dimension
    and :math:`m` is the dimension of the solution vector when used in the
    automatic generation of DG FE formulations by homogenisation.

    Parameters
    ----------
    G
        Homogeneity tensor
    tau
        Tensor to be multiplied

    Returns
    -------
    G^T * tau
    """
    if len(G.ufl_shape) == 0:
        if not len(tau.ufl_shape) == 0:
            raise IndexError("G^T is scalar, tau has shape: %s"
                             + str(tau.ufl_shape))
        return G*tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return dot(G.T, tau)
    elif ufl.rank(tau) == 1:
        return dot(G.T, tau)
    m, d = tau.ufl_shape
    return as_matrix([[inner(G[:, :, i, k], tau) for k in range(d)]
                      for i in range(m)])


def dg_outer(*args):
    # TODO: ufl treats this as (u ⊗ v*). If dolfin_dg goes complex need to
    #  fix this
    return outer(*args)


def homogeneity_tensor(F_v, u, differential_operator=grad):
    r"""Generate a homogeneity tensor :math:`G(u)` with respect to a linear
    differential operator :math:`\mathcal{L}(u)` such that

    .. math::

        G = \frac
        {\partial \mathcal{F}^v(u, \mathcal{L}(u))}
        {\partial \mathcal{L}(u)}

    For example consider the Poisson problem where :math:`\mathcal{F}^v(u,
    \nabla u) = \nabla u`. The homogeneity tensor in this case
    :math:`G_{ij} = \delta_{ij}`

    >>> import dolfin_dg, ufl
    >>> element = ufl.FiniteElement("CG", ufl.triangle, 1)
    >>> u = ufl.Coefficient(element)
    >>> G = dolfin_dg.homogeneity_tensor(lambda u, grad_u: grad_u, u)
    >>> G = ufl.algorithms.apply_derivatives.apply_derivatives(G)
    >>> assert G == ufl.Identity(2)

    Parameters
    ----------
    F_v
        Two argument callable function returning flux tensor
    u
        Solution variable
    differential_operator
        Single argument callable returning formulation of :math:`\mathcal{L}(u)`

    Returns
    -------
    Homogeneity tensor G
    """
    if len(inspect.getfullargspec(F_v).args) < 2:
        raise TypeError("Function F_v must have at least 2 arguments, "
                        "(u, grad_u, *args, **kwargs)")

    diff_op_u = variable(differential_operator(u))
    tau = F_v(u, diff_op_u)
    return diff(tau, diff_op_u)


def _get_terminal_operand_coefficient(u):
    """Traverse the UFL DAG to find a terminal operand. This is used,
    for example, when `u` belongs to a mixed space and we need to find
    terminal variables against which to homogenise.

    Parameters
    ----------
    u
        UFL expression

    Returns
    -------
    The first terminal operand of the expression
    """
    if not isinstance(u, (ufl.Coefficient, ufl.Argument)):
        return _get_terminal_operand_coefficient(
            u.ufl_operands[0])
    return u


def _get_ufl_element_degree(u):
    """Assuming `u` is either a function space, coefficient or ufl indexed
    operand, traverse the UFL DAG to find a terminal operand and return its
    UFL element degree. This is useful for automatically determining a "default"
    penalty parameter which is a function of the approximating polynomial
    degree.

    Parameters
    ----------
    u
        UFL expression

    Returns
    -------
    First terminal operands approximating polynomial degree
    """
    if isinstance(u, (ufl.Coefficient, ufl.FunctionSpace, ufl.Argument)):
        return u.ufl_element().degree()
    if isinstance(u, ufl.tensors.ListTensor):
        return _get_ufl_element_degree(_get_terminal_operand_coefficient(u))
    else:
        coeff, idx = u.ufl_operands
        element = coeff.ufl_element()
        return element.degree()


def _get_ufl_list_tensor_indices(u):
    if isinstance(u, ufl.tensors.ListTensor):
        return list(map(_get_ufl_list_tensor_indices, u.ufl_operands))
    return u.ufl_operands[1][0]._value


def generate_default_sipg_penalty_term(u, C_IP=20.0):
    r"""Generate a default and typically robust penalty parameter
    :math:`\sigma_\text{IP}` such that

    .. math::

        \sigma = C_\text{IP} \frac{\max (p^2, 1)}{h_\kappa}

    where :math:`C_\text{IP}` is the interior penalty parameter (independent
    of the mesh), :math:`p` is the polynomial degree of `u` and
    :math:`h_\kappa` is the cell diameter of cell :math:`\kappa`.

    Parameters
    ----------
    u
        FE solution variable
    C_IP
        Interior penalty parameter

    Returns
    -------
    Penalty parameter

    Warnings
    --------
    The penalty parameter generated by this function may not be appropriate
    and the user should craft something specific to their problem.
    """
    h = ufl.CellDiameter(u.ufl_domain())
    ufl_degree = _get_ufl_element_degree(u)
    return C_IP * max(ufl_degree ** 2, 1) / h


class DGFemTerm(abc.ABC):
    r"""Abstract base class for automatic generation of DG FE formulations for
    problems of the form

    .. math::

        - \nabla \cdot \mathcal{F}^v(u, \nabla u) = 0
    """

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n):
        """
        Parameters
        ----------
        F_v
            Two argument flux function `F_v(u, grad_u)`
        u_vec
            FE solution vector
        v_vec
            FE test function
        sigma
            Penalty parameter
        G
            Homogeneity tensor
        n
            Mesh facet normal
        """
        self.F_v = F_v
        self.U = u_vec
        self.V = v_vec
        self.sigma = sigma
        self.G = G
        self.n = n

    def _make_boundary_G(self, G, u_gamma):
        r"""Substitutes FE solution vector in the homogeneity tensor with the
        boundary flux function :math:`u_\gamma`. Necessary for formulation of
        boundary
        conditions.

        Parameters
        ----------
        G
            Homogeneity tensor
        u_gamma
            Boundary flux function

        Returns
        -------
        :math:`G(u_\Gamma)`
        """
        U = self.U
        U_soln = _get_terminal_operand_coefficient(U)

        # Reshape u_gamma if U_soln is from x mixed space
        bc_shape = u_gamma.ufl_shape[0] if u_gamma.ufl_shape else 1
        soln_shape = U_soln.ufl_shape[0] if U_soln.ufl_shape else 1
        if soln_shape > bc_shape:
            U_soln_idx = _get_ufl_list_tensor_indices(U)
            if not hasattr(U_soln_idx, "__len__"):
                U_soln_idx = (U_soln_idx,)

            # Construct new u_gamma from existing U_soln and replace
            # bc components with their prescribed data
            u_gamma_vec = [u_sub for u_sub in U_soln]
            if not u_gamma.ufl_shape:
                u_gamma = [u_gamma]
            for j, idx in enumerate(U_soln_idx):
                u_gamma_vec[idx] = u_gamma[j]
            u_gamma = ufl.as_vector(u_gamma_vec)

        if isinstance(G, ufl.core.expr.Expr):
            return replace(G, {U_soln: u_gamma})

        assert(isinstance(G, dict))
        G_gamma = {}
        for idx, tensor in G.items():
            G_gamma[idx] = replace(tensor, {U_soln: u_gamma})
        return G_gamma

    @abc.abstractmethod
    def interior_residual(self, dInt):
        """Automatically generated interior facet residual FE formulation

        Parameters
        ----------
        dInt
            Interior facet measure, e.g. `dS`

        Returns
        -------
        Interior residual formulation
        """
        pass

    @abc.abstractmethod
    def exterior_residual(self, u_gamma, dExt):
        """Automatically generated exterior facet residual FE formulation with
        weak imposition of boundary data

        Parameters
        ----------
        u_gamma
            Boundary flux function
        dExt
            Exterior facet measure, e.g. `ds`

        Returns
        -------
        Exterior residual formulation
        """
        pass

    @abc.abstractmethod
    def exterior_residual_on_interior(self, u_gamma, dExt):
        """
        Automatically generated exterior facet residual FE formulation with weak
        imposition of boundary data on an interior facet measure

        Parameters
        ----------
        u_gamma
            Boundary flux function
        dExt
            Interior facet measure, e.g. `dS`

        Returns
        -------
        Exterior residual formulation defined on interior boundary measure
        """
        pass

    @abc.abstractmethod
    def neumann_residual(self, g_N, dExt):
        """
        Parameters
        ----------
        g_N
            Neumann boundary data
        dExt
            Exterior boundary measure, e.g. `ds`

        Returns
        -------
        Neumann boundary formulation
        """
        return -inner(g_N, self.V)*dExt


class DGClassicalSecondOrderDiscretisation(DGFemTerm):
    r"""Implementation for automatic generation of DG FE formulations for
    problems of the form

    .. math::

        - \nabla \cdot \mathcal{F}^v(u, \nabla u) = 0
    """

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n, delta):
        super().__init__(F_v, u_vec, v_vec, sigma, G, n)
        self.delta = delta

    def interior_residual(self, dInt):
        G = self.G
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(
            tensor_jump(u, n), avg(hyper_tensor_T_product(G, grad_v))) * dInt \
            - inner(avg(self.F_v(u, grad_u)), tensor_jump(v, n)) * dInt
        if sigma is not None:
            residual += inner(
                sigma('+') * hyper_tensor_product(avg(G), tensor_jump(u, n)),
                tensor_jump(v, n)) * dInt

        residual = apply_dg_operators(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(
            dg_outer(u - u_gamma, n), hyper_tensor_T_product(G, grad_v)) \
            - inner(self.F_v(u, grad_u), dg_outer(v, n))
        if sigma is not None:
            residual += inner(
                sigma * hyper_tensor_product(G, dg_outer(u - u_gamma, n)),
                dg_outer(v, n))
        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))

    def neumann_residual(self, g_N, dExt):
        return -inner(g_N, self.V)*dExt


class DGFemSIPG(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = -1
        super().__init__(F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemNIPG(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = +1
        super().__init__(F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemBO(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = +1
        sigma = None
        super().__init__(F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemCurlTerm(DGFemTerm):
    r"""Implementation for automatic generation of DG FE formulations for
    problems of the form

    .. math::

        \nabla \times \mathcal{F}^m(u, \nabla \times u) = 0
    """

    def interior_residual(self, dInt):
        G = self.G
        u, v = self.U, self.V
        curl_u, curl_v = curl(u), curl(v)
        sigma, n = self.sigma, self.n

        residual = - inner(tangent_jump(u, n),
                           avg(hyper_tensor_T_product(G, curl_v)))*dInt \
            - inner(tangent_jump(v, n), avg(self.F_v(u, curl_u)))*dInt \
            + sigma('+')*inner(hyper_tensor_product(avg(G), tangent_jump(u, n)),
                               tangent_jump(v, n))*dInt

        residual = apply_dg_operators(residual)
        return residual

    def exterior_residual(self, u_gamma, dExt):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        curl_u, curl_v = curl(u), curl(v)
        sigma, n = self.sigma, self.n

        residual = - inner(dg_cross(n, u - u_gamma),
                           hyper_tensor_T_product(G, curl_v))*dExt \
            - inner(dg_cross(n, v),
                    hyper_tensor_product(G, curl_u))*dExt \
            + sigma*inner(hyper_tensor_product(G, dg_cross(n, u - u_gamma)),
                          dg_cross(n, v))*dExt
        return residual

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return NotImplementedError

    def neumann_residual(self, g_N, dExt):
        return NotImplementedError


class DGFemStokesTerm(DGClassicalSecondOrderDiscretisation):
    r"""Implementation for automatic generation of DG FE formulations for
    problems of the form

    .. math::

        - \nabla \cdot \mathcal{F}^v(u, \nabla u) &= 0 \\
        \nabla \cdot u &= 0
    """

    def __init__(self, F_v, u, p, v, q, sigma, G, n, delta,
                 block_form=False):
        self.u, self.v = u, v
        self.p, self.q = p, q
        self.block_form = block_form
        super().__init__(F_v, u, v, sigma, G, n, delta)

    def interior_residual(self, dInt):
        u = self.u
        q = self.q
        n = self.n

        residual = [super().interior_residual(dInt),
                    -ufl.jump(u, n) * avg(q) * dInt]
        residual = list(map(apply_dg_operators, residual))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        u = self.u
        q = self.q
        n = self.n

        residual = [super()._exterior_residual_no_integral(u_gamma),
                    - dot(u - u_gamma, n) * q]
        return residual

    def exterior_residual(self, u_gamma, dExt):
        residual = list(map(lambda Fj: Fj*dExt,
                            self._exterior_residual_no_integral(u_gamma)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def exterior_residual_on_interior(self, u_gamma, dExt):
        residual = list(map(
            lambda Fj: sum(Fj(side)*dExt for side in ("+", "-")),
            self._exterior_residual_no_integral(u_gamma)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def _slip_exterior_residual_no_integral(self, u_gamma, f2, tau=None):
        q = self.q

        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        if tau is None:
            tau = n

        # Velocity block
        F0 = delta * inner(
            u - u_gamma, normal_proj(hyper_tensor_T_product(G, grad_v) * n, tau)) \
             - inner(normal_proj(self.F_v(u, grad_u) * n, tau), v)
        if sigma is not None:
            F0 += inner(sigma * normal_proj(
                hyper_tensor_product(G, dg_outer(u - u_gamma, n)) * n, tau), v)

        # Tangential force
        F0 -= dot(tangential_proj(f2, tau), v)

        # Continuity block
        F1 = - dot(u - u_gamma, dolfin_dg.normal_proj(n, tau)) * q

        return [F0, F1]

    def slip_exterior_residual(self, u_gamma, f2, dExt, tau=None):
        residual = list(map(
            lambda Fj: Fj*dExt,
            self._slip_exterior_residual_no_integral(u_gamma, f2, tau=tau)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def slip_exterior_residual_on_interior(self, u_gamma, f2, dExt, tau=None):
        residual = list(map(
            lambda Fj: sum(Fj(side)*dExt for side in ("+", "-")),
            self._slip_exterior_residual_no_integral(u_gamma, f2, tau=tau)))
        if not self.block_form:
            residual = sum(residual)
        return residual


class DGClassicalFourthOrderDiscretisation(DGFemTerm):
    r"""Implementation for automatic generation of DG FE formulations for
    problems of the form

    .. math::

        - \nabla^4 u = 0
    """

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n, delta):
        super().__init__(F_v, u_vec, v_vec, sigma, G, n)
        self.delta = delta

    def interior_residual(self, dInt):
        G = self.G
        u, v = self.U, self.V
        grad_v = grad(v)
        div_grad_v = div(grad_v)
        grad_u = grad(u)
        div_grad_u = div(grad_u)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(
            jump(grad_u, n), avg(hyper_tensor_T_product(G, div_grad_v))) * dInt\
            - inner(avg(self.F_v(u, div_grad_u)), jump(grad_v, n)) * dInt
        if sigma is not None:
            residual += inner(
                sigma('+') * hyper_tensor_product(avg(G), jump(grad_u, n)),
                jump(grad_v, n)) * dInt

        residual = apply_dg_operators(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        div_grad_u, div_grad_v = div(grad_u), div(grad_v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(dot(grad(u - u_gamma), n),
                                 hyper_tensor_T_product(G, div_grad_v)) \
            - inner(self.F_v(u, div_grad_u), dot(grad_v, n))
        if sigma is not None:
            residual += inner(
                sigma * hyper_tensor_product(G, dot(grad(u - u_gamma), n)),
                dot(grad_v, n))
        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))

    def neumann_residual(self, g_N, dExt):
        raise NotImplementedError
