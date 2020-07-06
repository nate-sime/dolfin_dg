import dolfin


class NewtonSolverStaticCondensation(dolfin.NewtonSolver):

    def __init__(self, comm=dolfin.MPI.comm_world):
        dolfin.NewtonSolver.__init__(self, comm,
                                     dolfin.PETScKrylovSolver(),
                                     dolfin.PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        a


# Use a Newton solver with custom damping parameter
class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        self.linear_solver().set_from_options()

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem, iteration):
        tau = 1.0
        theta = min(sqrt(2.0 * tau / norm(dx, norm_type="l2", mesh=V.mesh())), 1.0)
        info("Newton damping parameter: %.3e" % theta)
        x.axpy(-theta, dx)