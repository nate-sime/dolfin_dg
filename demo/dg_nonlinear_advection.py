from dolfin import *
import numpy as np

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

n_eles = [4, 8, 16, 32, 64]
errorl2 = np.array([0]*len(n_eles), dtype=np.double)
errorh1 = np.array([0]*len(n_eles), dtype=np.double)
hsizes = np.array([0]*len(n_eles), dtype=np.double)

run_count = 0
for n_ele in n_eles:
    mesh = UnitSquareMesh(n_ele, n_ele, 'left/right')
    V = FunctionSpace(mesh, 'DG', 2)
    u = Function(V)
    v = TestFunction(V)
    
    n = FacetNormal(mesh)
    
    b = Constant((1.0, 1.0))
    c = Constant(1.0)
    f = Expression('exp(x[0] - x[1])')
    gD = f
    
    def F_c(U):
        return b*(U+1)**2
    
    def H(U_p, U_m, n_p):
        alpha = 2.0*Max(abs((U_p+1) * dot(b, n_p)), abs((U_m + 1) * dot(b, -n_p)))
        value = Constant(0.5)*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + alpha*(U_p - U_m))
        return value
    
    volume = -inner(F_c(u), grad(v))*dx + c*u*v*dx - f*v*dx
    interior = dot(H(u('+'), u('-'), n('+')), (v('+') - v('-')))*dS
    exterior = dot(H(u, gD, n), v)*ds
    
    residual = volume + interior + exterior
    
    du = TrialFunction(V)
    J = derivative(residual, u, du)
    
    solve(residual == 0, u, J=J)
    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()
    run_count += 1

if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    print np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:])
    print np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:])
