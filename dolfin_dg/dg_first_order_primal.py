import inspect
import abc

import ufl
from ufl import as_matrix, outer, inner, replace, grad, variable, diff, dot, curl, div

from dolfin_dg.dg_form import hyper_tensor_product, hyper_tensor_T_product, dg_outer, DGFemTerm
from dolfin_dg.dg_ufl import apply_dg_operator_lowering, avg, tensor_jump, jump, tangent_jump, dg_cross


class PrimalHomogeneityTensor:

    def __init__(self, operator):
        self.diff_op = operator

    def homogeneity_tensor(self, F_v, u):
        if len(inspect.getfullargspec(F_v).args) < 2:
            raise TypeError("Function F_v must have at least 2 arguments, "
                            "(u, grad_u, *args **kwargs)")

        diff_op_u = variable(self.diff_op(u))
        tau = F_v(u, diff_op_u)
        return diff(tau, diff_op_u)


class DGFOPrimalForm(DGFemTerm):

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n, number_ibp):
        self.number_ibp = number_ibp
        super().__init__(F_v, u_vec, v_vec, sigma, G, n)

    @abc.abstractmethod
    def flux_approximation(self):
        pass

    @abc.abstractmethod
    def flux_approximation_exterior(self, u_gamma):
        pass


class DGDivIBP(DGFOPrimalForm):

    def interior_residual(self, dInt):
        sigma, v = self.U, self.V
        n = self.n

        sigma_hat = self.flux_approximation()

        if self.number_ibp == 2:
            residual = - inner(avg(sigma_hat - sigma), tensor_jump(v, n)) * dInt
            residual += - inner(jump(sigma_hat - sigma, n), avg(v)) * dInt
        else:
            residual = - inner(avg(sigma_hat), tensor_jump(v, n)) * dInt
            residual += - inner(jump(sigma_hat, n), avg(v)) * dInt

        residual = apply_dg_operator_lowering(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        sigma, v = self.U, self.V

        n = self.n

        sigma_hat = self.flux_approximation_exterior(u_gamma)

        if self.number_ibp == 2:
            residual = - inner(sigma_hat - sigma, dg_outer(v, n))
        else:
            residual = - inner(sigma_hat, dg_outer(v, n))

        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))

    def neumann_residual(self, g_N, dExt):
        return -inner(g_N, self.V)*dExt


class DGDivIBP_SIPG(DGDivIBP):

    def flux_approximation(self):
        sigma = self.U
        penalty = self.sigma
        n = self.n
        u = self.temp_u
        sigma_hat = avg(sigma) - penalty("+") * tensor_jump(u, n)
        return sigma_hat

    def flux_approximation_exterior(self, u_gamma):
        sigma = self.U
        penalty = self.sigma
        n = self.n
        u = self.temp_u
        sigma_hat = sigma - penalty * dg_outer(u - u_gamma, n)
        return sigma_hat


class DGGradIBP(DGFOPrimalForm):

    def interior_residual(self, dInt):
        G = self.G
        u, tau = self.U, self.V
        n = self.n

        G = G.homogeneity_tensor(self.F_v, u)

        u_hat = self.flux_approximation()

        if self.number_ibp == 2:
            residual = inner(tensor_jump(u_hat - u, n), avg(hyper_tensor_T_product(G, tau))) * dInt
            residual += inner(avg(u_hat - u), jump(hyper_tensor_T_product(G, tau), n)) * dInt
        else:
            residual = inner(tensor_jump(u_hat, n), avg(hyper_tensor_T_product(G, tau))) * dInt
            residual += inner(avg(u_hat), jump(hyper_tensor_T_product(G, tau), n)) * dInt

        residual = apply_dg_operator_lowering(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        u, tau = self.U, self.V
        G = self.G.homogeneity_tensor(self.F_v, u)
        G = self._make_boundary_G(G, u_gamma)
        n = self.n

        u_hat = self.flux_approximation_exterior(u_gamma)

        if self.number_ibp == 2:
            residual = inner(dg_outer(u_hat - u, n), hyper_tensor_T_product(G, tau))
        else:
            residual = inner(dg_outer(u_hat, n), hyper_tensor_T_product(G, tau))

        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))

    def neumann_residual(self, g_N, dExt):
        return -inner(g_N, self.V)*dExt


class DGGradIBP_SIPG(DGGradIBP):

    def flux_approximation(self):
        u_hat = avg(self.U)
        return u_hat

    def flux_approximation_exterior(self, u_gamma):
        u_hat = u_gamma
        return u_hat

#
#
# class DGCurlIBP(DGFemTerm):
#
#     def __init__(self, F_v, u_vec, v_vec, sigma, G, n, delta):
#         super().__init__(F_v, u_vec, v_vec, sigma, G, n)
#         self.delta = delta
#         self.diff_op_test = curl
#
#     def _eval_F_v(self, U, grad_U):
#         assert len(inspect.getfullargspec(self.F_v).args) > 1
#         tau = self.F_v(U, grad_U)
#         return tau
#
#     def interior_residual(self, dInt):
#         G = self.G
#         u, v = self.U, self.V
#         diff_op_u, diff_op_v = G.diff_op(u), self.diff_op_test(v)
#         sigma, n = self.sigma, self.n
#         delta = self.delta
#
#         G = G.homogeneity_tensor(self.F_v, u)
#         residual = - inner(tangent_jump(u, n), avg(hyper_tensor_T_product(G, diff_op_v))) * dInt \
#                    - inner(tangent_jump(v, n), avg(self._eval_F_v(u, diff_op_u))) * dInt \
#                    + sigma('+') * inner(hyper_tensor_product(avg(G), tangent_jump(u, n)),
#                                         tangent_jump(v, n)) * dInt
#
#         return residual
#
#     def _exterior_residual_no_integral(self, u_gamma):
#         u, v = self.U, self.V
#         # diff_op_u, diff_op_v = self.G.diff_op(u), self.diff_op_test(v)
#         diff_op_u, diff_op_v = curl(u), self.diff_op_test(v)
#         G = self.G.homogeneity_tensor(self.F_v, u)
#         G = self._make_boundary_G(G, u_gamma)
#         sigma, n = self.sigma, self.n
#         delta = self.delta
#
#         residual = - inner(dg_cross(n, u - u_gamma), hyper_tensor_T_product(G, diff_op_v)) \
#                    - inner(dg_cross(n, v), hyper_tensor_product(G, diff_op_u)) \
#                    + sigma*inner(hyper_tensor_product(G, dg_cross(n, u - u_gamma)), dg_cross(n, v))
#         return residual
#
#     def exterior_residual(self, u_gamma, dExt):
#         return self._exterior_residual_no_integral(u_gamma) * dExt
#
#     def exterior_residual_on_interior(self, u_gamma, dExt):
#         return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
#                    for side in ("+", "-"))
#
#     def neumann_residual(self, g_N, dExt):
#         return NotImplementedError
#         # return -inner(g_N, self.V)*dExt
