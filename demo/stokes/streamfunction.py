import numpy as np

from dolfin import *
import dolfin_dg as dg

# C0-IPG formulation of the streamfunction formulation of the Stokes
# equations. Here we seek u such that
#
# -∇² u + ∇ p = f in Ω
#       u ⋅ n = 0 on ∂Ω
#
# Which may be reformulated in terms of the stream function u = ∇ × ψ
# so we seek ψ such that
#
#    ∇⁴ ψ = ∇ × f       in Ω
#       ψ = 0           on ∂Ω
# ∇ ψ ⋅ n = ∇ ψₛₒₗₙ ⋅ n on ∂Ω
#
# Where the manufactured solution
#
#  - ∇² ψₛₒₗₙ = ∇ × uₛₒₗₙ in Ω
#           ψ = 0         on ∂Ω
#
# We weakly enforce the C1 continuity by applying a DG method to the
# second integration by parts of the biharmonic operator.

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

ele_ns = [4, 8, 16, 32]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
errorpsil2 = np.zeros(len(ele_ns))
errorpsih1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))

p = 2

for j, n_ele in enumerate(ele_ns):
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), n_ele, n_ele, "left/right")
    PHI = FunctionSpace(mesh, "CG", p)
    V = VectorFunctionSpace(mesh, "CG", p-1)

    n = FacetNormal(mesh)
    psi = Function(PHI)
    phi = TestFunction(PHI)

    # Velocity solution where u ⋅ n = 0 on the boundary
    u_soln = Expression(("2*x[1]*(1.0 - x[0]*x[0])",
                         "-2*x[0]*(1.0 - x[1]*x[1])"),
                        degree=V.ufl_element().degree() + 2,
                        domain=mesh)

    # Compute the approximation of the true solution of ψ
    zero_bc = [DirichletBC(PHI, Constant(0.0), "on_boundary")]
    psi_soln = Function(PHI)
    F_psi = dot(grad(psi_soln), grad(phi)) * dx - curl(u_soln) * phi * dx
    solve(F_psi == 0, psi_soln, zero_bc)

    # The second order viscous flux
    def F_v(u, div_grad_u):
        return div_grad_u

    # Fourth order DG discretisation
    G = dg.homogeneity_tensor(F_v, psi, differential_operator=lambda u: div(grad(u)))
    sigma = dg.generate_default_sipg_penalty_term(psi, C_IP=Constant(1e1))
    fo = dg.DGClassicalFourthOrderDiscretisation(F_v, psi, phi, sigma, G, n, -1)

    # Finite element residual
    f = curl(-div(grad(u_soln)))
    F = inner(div(grad(psi)), div(grad(phi))) * dx - f * phi * dx
    F += fo.interior_residual(dS)

    # Weakly enforce ∇ ψ ⋅ n = ∇ ψₛₒₗₙ ⋅ n on ∂Ω
    F += fo.exterior_residual(psi_soln, ds)

    # Solve the linear system
    J = derivative(F, psi)
    solve(J == -F, psi, zero_bc)

    # Project the streamfunction to the (p-1) velocity space
    u = project(curl(psi), V)

    # Compute error
    errorl2[j] = errornorm(u_soln, u, norm_type='l2', degree_rise=3)
    errorh1[j] = errornorm(u_soln, u, norm_type='h1', degree_rise=3)
    errorpsil2[j] = errornorm(psi_soln, psi, norm_type='l2', degree_rise=3)
    errorpsih1[j] = errornorm(psi_soln, psi, norm_type='h1', degree_rise=3)
    hsizes[j] = mesh.hmax()


if MPI.rank(mesh.mpi_comm()) == 0:
    print("L2 u convergence rates: "
          + str(np.log(errorl2[0:-1] / errorl2[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
    print("H1 u convergence rates: "
          + str(np.log(errorh1[0:-1] / errorh1[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
    print("L2 psi convergence rates: "
          + str(np.log(errorpsil2[0:-1] / errorpsil2[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))
    print("H1 psi convergence rates: "
          + str(np.log(errorpsih1[0:-1] / errorpsih1[1:])
                / np.log(hsizes[0:-1] / hsizes[1:])))