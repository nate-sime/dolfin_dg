from dolfin import *
from dolfin_dg import *
import numpy as np

# WARNING: This demo is *very* computationally expensive to run.

parameters["ghost_mode"] = "shared_facet"
parameters["std_out_all_processes"] = False
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']["quadrature_degree"] = 4


class Problem(NonlinearProblem):
    def __init__(self, a, L, bcs):
        self.a = a
        self.L = L
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        assemble(self.a, tensor=A)


class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(), PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_rtol", 1e-3)

        self.linear_solver().set_from_options()


run_count = 0
ele_ns = [8, 16]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 1

for ele_n in ele_ns:
    # Mesh and function space.
    mesh = BoxMesh(Point(0, 0, 0), Point(pi, pi, pi), ele_n, ele_n, ele_n)
    V = VectorFunctionSpace(mesh, 'DG', p, dim=5)
    info("dofs: " + str(V.dim()))

    # Set up Dirichlet BC
    gD = Expression(('sin(2*(x[0]+x[1] + x[2])) + 4',
                     '0.2*sin(2*(x[0]+x[1] + x[2])) + 4',
                     '0.2*sin(2*(x[0]+x[1] + x[2])) + 4',
                     '0.2*sin(2*(x[0]+x[1] + x[2])) + 4',
                     'pow((sin(2*(x[0]+x[1] + x[2])) + 4), 2)'),
                    element=V.ufl_element())

    f = Expression(("1.2*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])",
                    "-2*mu*(8.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 16.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.6*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) - 3.2*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)) - 6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.4*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) + (0.4*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 1.6)*(6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.2*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.0*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])) + 0.8*(-1.5*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])",
                    "-2*mu*(8.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 16.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.6*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) - 3.2*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)) - 6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.4*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) + (0.4*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 1.6)*(6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.2*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.0*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])) + 0.8*(-1.5*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])",
                    "-2*mu*(8.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 16.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.6*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) - 3.2*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)) - 6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.4*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) + (0.4*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 1.6)*(6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.2*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.0*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])) + 0.8*(-1.5*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])",
                    "12.0*mu*(-4.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 0.8*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0))*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) - 2.4*mu*(-4.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 0.8*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0))*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) - 6*mu*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*(8.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 16.0*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.6*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) - 3.2*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2))/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) + 3*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*((0.4*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 1.6)*(6.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 1.2*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.0*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]))/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) - 2.0*(0.4*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 1.6)*(-1.5*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 2.0*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 0.8*(-1.5*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)) + 1.2*((0.4*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 1.6)*(-1.5*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)/(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0) + sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) - 3*gamma*mu*(-12.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 36.0*pow(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2)*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 4) + 2.4*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2])/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2) + 9.6*(0.2*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0)*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 3) - 4.0*sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) - 0.48*pow(cos(2.0*x[0] + 2.0*x[1] + 2.0*x[2]), 2)/pow(sin(2.0*x[0] + 2.0*x[1] + 2.0*x[2]) + 4.0, 2))/Pr",
                    ),
                   mu=1.0, Pr=0.72, gamma=1.4, element=V.ufl_element())

    u, v = interpolate(gD, V), TestFunction(V)
    du = TrialFunction(V)

    bo = CompressibleNavierStokesOperator(mesh, V, DGDirichletBC(ds, gD), mu=1.0)
    residual = bo.generate_fem_formulation(u, v) - inner(f, v)*dx
    J = derivative(residual, u, du)

    solver = CustomSolver()
    problem = Problem(J, residual, [])
    solver.solve(problem, u.vector())

    errorl2[run_count] = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1[run_count] = errornorm(gD, u, norm_type='h1', degree_rise=3)
    hsizes[run_count] = mesh.hmax()

    run_count += 1

if MPI.rank(mesh.mpi_comm()) == 0:
    print(np.log(errorl2[0:-1]/errorl2[1:])/np.log(hsizes[0:-1]/hsizes[1:]))
    print(np.log(errorh1[0:-1]/errorh1[1:])/np.log(hsizes[0:-1]/hsizes[1:]))