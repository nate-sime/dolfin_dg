from ufl import as_vector, dot


def conserved_variables(U):
    rho, rhoE = U[0], U[-1]
    dim = U.ufl_shape[0] - 2
    rhou = as_vector([U[j] for j in range(1, dim + 1)])
    return rho, rhou, rhoE


def flow_variables(U):
    rho, rhou, rhoE = conserved_variables(U)
    return rho, rhou/rho, rhoE/rho


def pressure(U, gamma=1.4):
    rho, u, E = flow_variables(U)
    p = (gamma - 1.0)*rho*(E - 0.5*dot(u, u))
    return p


def enthalpy(U, gamma=1.4):
    rho, E = U[0], U[-1]/U[0]
    p = pressure(U, gamma)
    H = E + p/rho
    return H


def speed_of_sound(p, rho, gamma=1.4):
    return abs(gamma*p/rho)**0.5


def effective_reynolds_number(Re_0, M_0, gamma=1.4):
    return Re_0/(gamma**0.5*M_0)


def energy_density(p, rho, u, gamma=1.4):
    return p/(gamma-1.0) + 0.5*rho*dot(u, u)


def subsonic_inflow(rho_in, u_in, u_vec):
    p = pressure(u_vec)
    rhoE_in = energy_density(p, rho_in, u_in)
    return as_vector([rho_in] + [rho_in*u_in[j] for j in range(u_in.ufl_shape[0])] + [rhoE_in])


def subsonic_outflow(p_out, u_vec):
    rho, u, E = flow_variables(u_vec)
    rhoE_out = energy_density(p_out, rho, u)
    dim = u_vec.ufl_shape[0] - 2
    return as_vector([rho] + [u_vec[j] for j in range(1, dim+1)] + [rhoE_out])


def no_slip(U):
    rho, rhou, rhoE = conserved_variables(U)
    dim = U.ufl_shape[0] - 2
    return as_vector([rho] + [0]*dim + [rhoE])
