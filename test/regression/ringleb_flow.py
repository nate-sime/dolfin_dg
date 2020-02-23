import numpy as np
import ufl

from dolfin import *

from dolfin_dg import force_zero_function_derivative
from dolfin_dg.dolfin import ringleb

__author__ = 'njcs4'


parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 10


class RinglebAnalSoln(UserExpression):
    def eval(self, value, x):
        val = ringleb.ringleb_anal_soln(x[0], x[1])
        value[0] = val[0]
        value[1] = val[1]
        value[2] = val[2]
        value[3] = val[3]

    def value_shape(self):
        return (4,)


def get_bdry_no(x0, x1):
    tol = 1.0e-6

    b_nos = [0, 0]
    x = [x0, x1]

    gamma = 1.4
    gammam1 = gamma-1

    psimax = 1.0/0.6
    psimin = 1.0/0.98
    qmin   = 0.43

    for j in range(2):
        c = ringleb.cspeed_ringleb(gamma, x[j][0], x[j][1])

        jval  = 1.0/c + 1.0/(3.0*c**3) + 1.0/(5.0*c**5) - 0.5*np.log( (1.0+c)/(1.0-c) )
        rho   = c**(2.0/gammam1)
        q2    = 2.0*(1.0-c**2)/gammam1
        bigv  = sqrt(q2)
        kval  = sqrt( 2.0/(1.0/q2 -2.0*rho*(x[j][0] - jval/2.0)))
        psi   = 1.0/kval

        if abs(psi-psimin) < tol or abs(psi-psimax) < tol:
            b_nos[j] = 1

    if b_nos[0] == b_nos[1]:
        return b_nos[0]
    return 0


poly_o = 1
run_count = 0
mesh_sizes = [3, 5, 9, 17, 33, 65]#, 128, 256]

mesh_cells = np.array([0]*len(mesh_sizes))
mesh_dofs = np.array([0]*len(mesh_sizes))

errorl2 = np.array([0]*len(mesh_sizes), dtype=np.double)
errorh1 = np.array([0]*len(mesh_sizes), dtype=np.double)
hsizes = np.array([0]*len(mesh_sizes), dtype=np.double)

for n_nodes in mesh_sizes:
    mesh = ringleb.ringleb_mesh(n_nodes, n_nodes, curved=False)
    # mesh = Mesh()
    # XDMFFile('ringleb_meshes/ringleb_mesh_%d_curved.xdmf' % n_nodes).read(mesh)

    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
    du = TrialFunction(V)
    v_vec = TestFunction(V)

    facet = MeshFunction('size_t', mesh, 1, 0)

    for f in facets(mesh):
        if f.exterior():
            v0 = Vertex(mesh, f.entities(0)[0]).midpoint()
            v1 = Vertex(mesh, f.entities(0)[1]).midpoint()
            facet[f] = get_bdry_no(v0, v1)

    ds = Measure('ds', subdomain_data=facet)

    gamma = 1.4
    gD = RinglebAnalSoln(element=V.ufl_element(), domain=mesh)
    u_vec = project(gD, V)
    n = FacetNormal(mesh)

    def F_c(U):
        rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
        p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
        H = E + p/rho
        res = as_matrix([[rho*u1, rho*u2],
                      [rho*u1**2 + p, rho*u1*u2],
                      [rho*u1*u2, rho*u2**2 + p],
                      [rho*H*u1, rho*H*u2]])
        return res

    convective_domain = -inner(F_c(u_vec), grad(v_vec))*dx

    def construct_evs(U, n):
        rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
        p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
        u = as_vector([u1, u2])
        c = sqrt(gamma*p/rho)
        lambdas = [dot(u,n) - c, dot(u,n), dot(u,n) + c]
        lambdas = list(map(abs, lambdas))
        return lambdas

    def construct_alpha(U_p, U_m, n_p):
        forward_evs = construct_evs(U_p, n_p)
        reverse_evs = construct_evs(U_m, n_p)

        return ufl.Max(
            ufl.Max(ufl.Max(forward_evs[0], forward_evs[1]), forward_evs[2]),
            ufl.Max(ufl.Max(reverse_evs[0], reverse_evs[1]), reverse_evs[2])
        )

    def H(U_p, U_m, n_p):
        U_p_a, U_m_a = force_zero_function_derivative(U_p), force_zero_function_derivative(U_m)
        alpha = construct_alpha(U_p_a, U_m_a, n_p)
        value = Constant(0.5)*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + Constant(1.0)*alpha*(U_p - U_m))
        return value

    convective_interior = dot(H(u_vec('+'), u_vec('-'), n('+')), (v_vec('+') - v_vec('-'))) * dS
    convective_exterior = dot(H(u_vec, gD, n), v_vec) * ds(0)

    # Slip wall
    no_slip_projector = as_matrix(((1,            0,            0, 0),
                                   (0,  1-2*n[0]**2, -2*n[0]*n[1], 0),
                                   (0, -2*n[0]*n[1],  1-2*n[1]**2, 0),
                                   (0,            0,            0, 1)))
    slip_bc = no_slip_projector * u_vec
    convective_exterior += dot(H(u_vec, slip_bc, n), v_vec)*ds(1)

    f = Constant((0.0, 0.0, 0.0, 0.0))

    residual = convective_domain + convective_interior + convective_exterior - dot(f, v_vec)*dx

    J = derivative(residual, u_vec, du)
    solve(residual == 0, u_vec, [], J=J)

    errorl2[run_count] = assemble(inner(gD - u_vec, gD - u_vec)*dx)**0.5
    errorh1[run_count] = assemble(inner(gD - u_vec, gD - u_vec)*dx + inner(grad(gD - u_vec), grad(gD - u_vec))*dx)**0.5
    hsizes[run_count] = mesh.hmax()
    mesh_cells[run_count] = mesh.num_cells()
    mesh_dofs[run_count] = V.dim()
    run_count += 1

flt = lambda x: "%.4e" % x

print('Elements', mesh_cells)
print('||u - u_h||_L2', list(map(flt, errorl2)))
print('||u - u_h||_H1', list(map(flt, errorh1)))
print('DoF', mesh_dofs)

if MPI.rank(mesh.mpi_comm()) == 0:
    print('k L2', np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print('k H1', np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
