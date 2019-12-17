from dolfin import *
import numpy as np

from dolfin_dg import StokesNitscheBoundary

parameters['std_out_all_processes'] = False
parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
errorpl2 = np.zeros(len(ele_ns))
errorph1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))

for j, ele_n in enumerate(ele_ns):
    mesh = UnitSquareMesh(ele_n, ele_n)
    We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
                       FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = FunctionSpace(mesh, We)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    u_soln = Expression(("-(x[1] * cos(x[1]) + sin(x[1])) * exp(x[0])",
                         "x[1] * sin(x[1]) * exp(x[0])"),
                        degree=4, domain=mesh)
    p_soln = Expression("2.0 * exp(x[0]) * sin(x[1]) + 1.5797803888225995912 / 3.0",
                        degree=4, domain=mesh)

    U = Function(W)
    u, p = split(U)

    V = TestFunction(W)
    v, q = split(V)

    ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain("near(x[0], 1.0) or near(x[1], 0.0)").mark(ff, 2)
    CompiledSubDomain("near(x[0], 0.0) or near(x[1], 1.0)").mark(ff, 1)
    CompiledSubDomain("near(x[0], x[1]) or near(x[0], -x[1])").mark(ff, 3)
    ds = Measure("ds", subdomain_data=ff)
    dsN, dsD = ds(2), ds(1)

    dS = Measure("dS", subdomain_data=ff)
    dSD = dS(3)

    def F_v(u, grad_u, p_local=None):
        if p_local is None:
            p_local = p
        return sym(grad_u) - p_local*Identity(2)

    f = -div(F_v(u_soln, grad(u_soln), p_soln))
    gN = F_v(u_soln, grad(u_soln), p_soln) * n
    res1 = inner(F_v(u, grad(u)), grad(v)) * dx - dot(gN, v) * dsN - dot(f, v) * dx
    res2 = div(u) * q * dx

    res = res1 + res2

    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q, delta=-1)
    res += stokes_nitsche.slip_nitsche_bc_residual(u_soln, gN, dsD)
    res += stokes_nitsche.slip_nitsche_bc_residual_on_interior(u_soln, gN, dSD)

    bcs = []
    solve(res == 0, U)

    errorl2[j] = errornorm(u_soln, U.sub(0), norm_type='l2', degree_rise=3)
    errorh1[j] = errornorm(u_soln, U.sub(0), norm_type='h1', degree_rise=3)
    errorpl2[j] = errornorm(p_soln, U.sub(1), norm_type='l2', degree_rise=3)
    errorph1[j] = errornorm(p_soln, U.sub(1), norm_type='h1', degree_rise=3)
    hsizes[j] = mesh.hmax()


if MPI.rank(mesh.mpi_comm()) == 0:
    print("L2 u convergence rates: " + str(np.log(errorl2[0:-1] / errorl2[1:]) / np.log(hsizes[0:-1] / hsizes[1:])))
    print("H1 u convergence rates: " + str(np.log(errorh1[0:-1] / errorh1[1:]) / np.log(hsizes[0:-1] / hsizes[1:])))
    print("L2 p convergence rates: " + str(np.log(errorpl2[0:-1] / errorpl2[1:]) / np.log(hsizes[0:-1] / hsizes[1:])))
    print("H1 p convergence rates: " + str(np.log(errorph1[0:-1] / errorph1[1:]) / np.log(hsizes[0:-1] / hsizes[1:])))
