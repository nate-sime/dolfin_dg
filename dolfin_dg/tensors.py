from ufl import as_matrix


__author__ = 'njcs4'


def compressible_ns_G(u_vec, mu, Pr, gamma):

    G = {}

    G[0, 0] = as_matrix([[0, 0, 0, 0],
                         [-4*mu*u_vec[1]/(3*u_vec[0]**2), 4*mu/(3*u_vec[0]), 0, 0],
                         [-mu*u_vec[2]/u_vec[0]**2, 0, mu/u_vec[0], 0],
                         [mu*(-Pr*(4*u_vec[1]**2 + 3*u_vec[2]**2) + 3*gamma*(-u_vec[0]*u_vec[3] + u_vec[1]**2 + u_vec[2]**2))/(3*Pr*u_vec[0]**3), mu*(4*Pr - 3*gamma)*u_vec[1]/(3*Pr*u_vec[0]**2), mu*(Pr - gamma)*u_vec[2]/(Pr*u_vec[0]**2), gamma*mu/(Pr*u_vec[0])]])

    G[0, 1] = as_matrix([[0, 0, 0, 0],
                         [2*mu*u_vec[2]/(3*u_vec[0]**2), 0, -2*mu/(3*u_vec[0]), 0],
                         [-mu*u_vec[1]/u_vec[0]**2, mu/u_vec[0], 0, 0],
                         [-mu*u_vec[1]*u_vec[2]/(3*u_vec[0]**3), mu*u_vec[2]/u_vec[0]**2, -2*mu*u_vec[1]/(3*u_vec[0]**2), 0]])

    G[1, 0] = as_matrix([[0, 0, 0, 0],
                         [-mu*u_vec[2]/u_vec[0]**2, 0, mu/u_vec[0], 0],
                         [2*mu*u_vec[1]/(3*u_vec[0]**2), -2*mu/(3*u_vec[0]), 0, 0],
                         [-mu*u_vec[1]*u_vec[2]/(3*u_vec[0]**3), -2*mu*u_vec[2]/(3*u_vec[0]**2), mu*u_vec[1]/u_vec[0]**2, 0]])

    G[1, 1] = as_matrix([[0, 0, 0, 0],
                         [-mu*u_vec[1]/u_vec[0]**2, mu/u_vec[0], 0, 0],
                         [-4*mu*u_vec[2]/(3*u_vec[0]**2), 0, 4*mu/(3*u_vec[0]), 0],
                         [mu*(-Pr*(3*u_vec[1]**2 + 4*u_vec[2]**2) + 3*gamma*(-u_vec[0]*u_vec[3] + u_vec[1]**2 + u_vec[2]**2))/(3*Pr*u_vec[0]**3), mu*(Pr - gamma)*u_vec[1]/(Pr*u_vec[0]**2), mu*(4*Pr - 3*gamma)*u_vec[2]/(3*Pr*u_vec[0]**2), gamma*mu/(Pr*u_vec[0])]])
    return G