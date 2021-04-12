import abc

import ufl

from dolfin_dg import hyper_tensor_product, hyper_tensor_T_product, \
    DGDirichletBC


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

    def face_residual(self, dInt, dExt, u_flux=None):
        G = self.G
        u, ubar = self.u, self.ubar
        v, vbar = self.v, self.vbar
        grad_u, grad_v = ufl.grad(u), ufl.grad(v)
        sigma, n = self.sigma, self.n

        # Redefine the facet integral for the provided measures
        def facet_int(integrand):
            return facet_integral(integrand, dInt, dExt)

        if u_flux is None:
            u_flux = ubar
        F_v_flux = self.F_v(u, grad_u) \
            + sigma * hyper_tensor_product(G, ufl.outer(u_flux - u, n))

        residual0 = facet_int(ufl.inner(ufl.outer(u_flux - u, n),
                                        hyper_tensor_T_product(G, grad_v))) \
            - facet_int(ufl.inner(F_v_flux, ufl.outer(v, n))) \

        residual1 = facet_int(ufl.inner(F_v_flux, ufl.outer(vbar, n)))

        residual = residual0 + residual1

        return residual

    def neumann_residual(self, g_N, dExt):
        return -ufl.inner(g_N, self.vbar)*dExt


class HDGClassicalFirstOrder:

    def __init__(self, F_c, u, ubar, v, vbar, H_flux, n, dg_bcs=[]):
        self.F_c = F_c
        self.u, self.ubar = u, ubar
        self.v, self.vbar = v, vbar
        self.H_flux = H_flux
        self.n = n
        self.dg_bcs = dg_bcs

    def face_residual(self, dInt, dExt):
        u, ubar = self.u, self.ubar
        v, vbar = self.v, self.vbar
        n = self.n

        # Redefine the facet integral for the provided measures
        def facet_int(integrand):
            return facet_integral(integrand, dInt, dExt)

        def interior_facet_int(integrand):
            return facet_integral(integrand, dInt, 0)

        import dolfin_dg.fluxes
        evs_u = self.H_flux.flux_jacobian_eigenvalues(u, n)
        if not isinstance(evs_u, (tuple, list)):
            evs_u = (evs_u,)
        evs_bar = self.H_flux.flux_jacobian_eigenvalues(ubar, n)
        if not isinstance(evs_bar, (tuple, list)):
            evs_bar = (evs_bar,)

        # `LF' flux
        S = dolfin_dg.fluxes.max_abs_of_sequence([*evs_u, *evs_bar])
        F_c_flux = 0.5*ufl.dot(self.F_c(u) + self.F_c(ubar), n) \
                   + S * (u - ubar)

        residual0 = facet_int(ufl.inner(F_c_flux, v))

        residual1 = -facet_int(ufl.inner(F_c_flux, vbar))

        for bc in self.dg_bcs:
            if not isinstance(bc, DGDirichletBC):
                    continue
            ds_bc = bc.get_boundary()
            gD = bc.get_function()

            evs_gD = self.H_flux.flux_jacobian_eigenvalues(gD, n)
            if not isinstance(evs_gD, (tuple, list)):
                evs_gD = (evs_gD,)
            S = dolfin_dg.fluxes.max_abs_of_sequence([*evs_u, *evs_gD])
            F_c_flux = 0.5 * ufl.dot(self.F_c(u) + self.F_c(gD), n) \
                       + S * (u - gD)

            residual1 += ufl.inner(F_c_flux, vbar)*ds_bc

        residual = residual0 + residual1

        # # Cockburn LF style
        # S = dolfin_dg.fluxes.max_abs_of_sequence([*evs_u, *evs_bar])
        #
        # F_c_flux = ufl.dot(self.F_c(ubar), n) + S * (u - ubar)
        #
        # residual0 = facet_int(ufl.inner(F_c_flux, v))
        #
        # residual1 = interior_facet_int(ufl.inner(F_c_flux, vbar))
        #
        # for bc in self.dg_bcs:
        #     if not isinstance(bc, DGDirichletBC):
        #         continue
        #     ds_bc = bc.get_boundary()
        #     gD = bc.get_function()
        #     evs_gD = self.H_flux.flux_jacobian_eigenvalues(gD, n)
        #     if not isinstance(evs_gD, (tuple, list)):
        #         evs_gD = (evs_gD,)
        #
        #     S = dolfin_dg.fluxes.max_abs_of_sequence([*evs_u, *evs_gD])
        #     B = S * (gD - ubar)
        #     residual1 += ufl.inner(B, vbar) * ds_bc
        #
        # residual = residual0 + residual1

        # # `Vijayasundaram' flux
        # max_evs = dolfin_dg.fluxes.max_of_sequence([*evs_u, *evs_bar, 0])
        # min_evs = dolfin_dg.fluxes.min_of_sequence([*evs_u, *evs_bar, 0])
        # S_p = max_evs
        # S_m = min_evs
        #
        # # interior
        # F_c_flux = S_p * u + S_m * ubar
        # residual0 = -facet_int(ufl.inner(F_c_flux, vbar - v))
        #
        # # exterior
        # residual1 = 0
        # for bc in self.dg_bcs:
        #     if not isinstance(bc, DGDirichletBC):
        #         continue
        #     ds_bc = bc.get_boundary()
        #     gD = bc.get_function()
        #
        #     evs_gD = self.H_flux.flux_jacobian_eigenvalues(gD, n)
        #     if not isinstance(evs_gD, (tuple, list)):
        #         evs_gD = (evs_gD,)
        #     S_m = dolfin_dg.fluxes.min_of_sequence([*evs_u, *evs_gD, 0])
        #     F_c_flux = S_p * ubar + S_m * gD
        #     residual1 += ufl.inner(F_c_flux, vbar)*ds_bc
        #
        # residual = residual0 + residual1

        return residual

    def neumann_residual(self, g_N, dExt):
        return -ufl.inner(g_N, self.vbar)*dExt
