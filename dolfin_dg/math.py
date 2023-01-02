import inspect

import ufl
from ufl import dot, as_matrix, inner, outer, grad, variable, diff


def dg_cross(u, v):
    if len(u.ufl_shape) == 0 and len(v.ufl_shape) == 0:
        raise TypeError("One input argument must be a vector")
    if len(u.ufl_shape) == 0 and len(v.ufl_shape) == 1:
        assert v.ufl_shape[0] == 2
        u = ufl.as_vector((0, 0, u))
        v = ufl.as_vector((v[0], v[1], 0))
        u_cross_v = ufl.cross(u, v)
        return ufl.as_vector((u_cross_v[0], u_cross_v[1]))
    if len(v.ufl_shape) == 0 and len(u.ufl_shape) == 1:
        assert u.ufl_shape[0] == 2
        u = ufl.as_vector((u[0], u[1], 0))
        v = ufl.as_vector((0, 0, v))
        u_cross_v = ufl.cross(u, v)
        return ufl.as_vector((u_cross_v[0], u_cross_v[1]))
    assert(len(u.ufl_shape) == 1 and len(v.ufl_shape) == 1)
    if u.ufl_shape[0] == 2 and v.ufl_shape[0] == 2:
        return u[0]*v[1] - u[1]*v[0]
    return ufl.cross(u, v)


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
        return G*tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return ufl.dot(G, tau.T).T
    elif ufl.rank(tau) == 1:
        return ufl.dot(G, tau)
    m, d = tau.ufl_shape
    return ufl.as_matrix([[ufl.inner(G[i, k, :, :], tau) for k in range(d)]
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
            raise IndexError(f"G^T is scalar, tau has shape: {tau.ufl_shape}")
        return G*tau
    elif ufl.rank(tau) == 0 and ufl.rank(G) == 2:
        return G.T * tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return ufl.dot(G.T, tau)
    elif ufl.rank(tau) == 1:
        return ufl.dot(G.T, tau)
    m, d = tau.ufl_shape
    return ufl.as_matrix([[ufl.inner(G[:, :, i, k], tau) for k in range(d)]
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