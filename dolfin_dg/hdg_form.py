import abc
import ufl

from dolfin_dg import hyper_tensor_product, hyper_tensor_T_product


def facet_integral(integrand, dS, ds):
    return integrand('-') * dS + integrand('+') * dS + integrand * ds


class HDGFemTerm(abc.ABC):

    def __init__(self, F_v, u, ubar, v, vbar, sigma, G, n):
        self.F_v = F_v
        self.u, self.ubar = u, ubar
        self.v, self.vbar = v, vbar
        self.sigma = sigma
        self.G = G
        self.n = n

    @abc.abstractmethod
    def face_residual(self, dInt, dExt):
        pass

    @abc.abstractmethod
    def neumann_residual(self, g_N, dExt):
        pass


class HDGClassicalSecondOrder(HDGFemTerm):

    def face_residual(self, dInt, dExt):
        G = self.G
        u, ubar = self.u, self.ubar
        v, vbar = self.v, self.vbar
        grad_u, grad_v = ufl.grad(u), ufl.grad(v)
        sigma, n = self.sigma, self.n

        # Redefine the facet integral for the provided measures
        facet_int = lambda integrand: facet_integral(integrand, dInt, dExt)

        u_flux = ubar
        F_v_flux = self.F_v(u, grad_u) + sigma * hyper_tensor_product(G, ufl.outer(u_flux - u, n))

        residual0 = facet_int(ufl.inner(ufl.outer(u_flux - u, n), hyper_tensor_T_product(G, grad_v))) \
                    - facet_int(ufl.inner(F_v_flux, ufl.outer(v, n))) \

        residual1 = facet_int(ufl.inner(F_v_flux, ufl.outer(vbar, n)))

        residual = residual0 + residual1

        return residual

    def neumann_residual(self, g_N, dExt):
        return -ufl.inner(g_N, self.vbar)*dExt


class HDGClassicalFirstOrder:

    def __init__(self, F_c, u, ubar, v, vbar, H_flux, n):
        self.F_c = F_c
        self.u, self.ubar = u, ubar
        self.v, self.vbar = v, vbar
        self.H_flux = H_flux
        self.n = n

    def face_residual(self, dInt, dExt):
        u, ubar = self.u, self.ubar
        v, vbar = self.v, self.vbar
        n = self.n

        # Redefine the facet integral for the provided measures
        facet_int = lambda integrand: facet_integral(integrand, dInt, dExt)

        S = abs(self.H_flux.flux_jacobian_eigenvalues(ubar, n))

        F_c_flux = ufl.dot(self.F_c(ubar), n) + S * (u - ubar)

        residual0 = facet_int(ufl.inner(F_c_flux, v))

        residual1 = -facet_int(ufl.inner(F_c_flux, vbar))

        residual = residual0 + residual1

        return residual

    def neumann_residual(self, g_N, dExt):
        return -ufl.inner(g_N, self.vbar)*dExt
