import matplotlib.pyplot as plt
import numpy as np
from dolfin import (
    UnitSquareMesh, FunctionSpace, Function, TestFunction, Expression,
    ds, dx, solve, parameters, errornorm, MPI, info)

from dolfin_dg import (DGFemSIPG, DGFemNIPG, DGFemBO, DGDirichletBC,
                       PoissonOperator)

if MPI.size(MPI.comm_world) > 1:
    NotImplementedError("Plotting in this demo will not work in parallel.")

parameters["ghost_mode"] = "shared_facet"

run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 2

# List of fluxes to be used in the comparison
fluxes = [("SIPG", DGFemSIPG),
          ("NIPG", DGFemNIPG),
          ("Baumann-Oden", DGFemBO)]

# Solve the linear Poisson problem and return the L2 and H1 errors, in addition
# to the mesh size.
def compute_error(ele_n, flux):
    mesh = UnitSquareMesh(ele_n, ele_n)

    V = FunctionSpace(mesh, 'DG', p)
    u, v = Function(V), TestFunction(V)

    gD = Expression('sin(pi*x[0])*sin(pi*x[1])',
                    element=V.ufl_element())
    f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])',
                   element=V.ufl_element())

    pe = PoissonOperator(mesh, V, DGDirichletBC(ds, gD))
    F = pe.generate_fem_formulation(u, v, vt=flux) - f*v*dx
    solve(F == 0, u)

    errorl2 = errornorm(gD, u, norm_type='l2', degree_rise=3)
    errorh1 = errornorm(gD, u, norm_type='h1', degree_rise=3)
    h_size = mesh.hmax()
    return (errorl2, errorh1, h_size)


# Compute rates of convergence
def compute_rate(errors, hsizes):
    return np.log(errors[0:-1]/errors[1:])/np.log(hsizes[0:-1]/hsizes[1:])


l2_plt = plt.figure(1).add_subplot(1, 1, 1)
h1_plt = plt.figure(2).add_subplot(1, 1, 1)
rate_messages = []
for name, flux in fluxes:
    error = np.array([compute_error(ele_n, flux) for ele_n in ele_ns])
    h_sizes = error[:, 2]

    l2_rates = compute_rate(error[:, 0], h_sizes)
    h1_rates = compute_rate(error[:, 1], h_sizes)

    rate_messages += [
        f"{name} flux\n"
        f"L2 rates: {str(l2_rates)}\n"
        f"H1 rates: {str(h1_rates)}"]

    l2_plt.loglog(h_sizes, error[:, 0], "-x")
    h1_plt.loglog(h_sizes, error[:, 1], "--x")


info("\n".join(rate_messages))

l2_plt.legend([name for (name, _) in fluxes])
l2_plt.set_ylabel("$\\Vert u - u_h \\Vert_{L_2}$")
l2_plt.set_xlabel("$h$")
l2_plt.set_title(f"$p = {p}$")

h1_plt.legend([name for (name, _) in fluxes])
h1_plt.set_ylabel("$\\Vert u - u_h \\Vert_{H_1}$")
h1_plt.set_xlabel("$h$")
h1_plt.set_title(f"$p = {p}$")

plt.show()
