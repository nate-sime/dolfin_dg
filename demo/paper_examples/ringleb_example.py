from dolfin import *
from dolfin_dg import *
from dolfin_dg import ringleb
import numpy as np

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 10


class RinglebAnalSoln(Expression):
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
        c = ringleb.cspeed_ringleb(gamma,x[j][0], x[j][1])

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
mesh_sizes = [3, 5, 9, 17, 33, 65]

errorl2 = np.array([0]*len(mesh_sizes), dtype=np.double)
errorh1 = np.array([0]*len(mesh_sizes), dtype=np.double)
hsizes = np.array([0]*len(mesh_sizes), dtype=np.double)

for n_nodes in mesh_sizes:
    mesh = ringleb.ringleb_mesh(n_nodes, n_nodes, curved=True)

    V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
    v_vec = TestFunction(V)

    facet = FacetFunction('size_t', mesh, 0)

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

    # Slip wall
    no_slip_projector = as_matrix(((1,            0,            0, 0),
                                   (0,  1-2*n[0]**2, -2*n[0]*n[1], 0),
                                   (0, -2*n[0]*n[1],  1-2*n[1]**2, 0),
                                   (0,            0,            0, 1)))
    slip_bc = no_slip_projector * u_vec

    bcs = [DGDirichletBC(ds(0), gD), DGDirichletBC(ds(1), slip_bc)]
    ceo = CompressibleEulerOperator(mesh, V, bcs)
    residual = ceo.generate_fem_formulation(u_vec, v_vec)

    solve(residual == 0, u_vec)

    errorl2[run_count] = assemble(inner(gD - u_vec, gD - u_vec)*dx)**0.5
    errorh1[run_count] = assemble(inner(gD - u_vec, gD - u_vec)*dx + inner(grad(gD - u_vec), grad(gD - u_vec))*dx)**0.5
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print 'k L2', np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print 'k H1', np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])
