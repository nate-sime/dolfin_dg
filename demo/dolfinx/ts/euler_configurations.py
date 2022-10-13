# ftp://ftp.math.ucla.edu/pub/camreport/cam00-34.pdf
from dolfin import *
from dolfin_dg import *

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 10

mesh = RectangleMesh.create(MPI.comm_world,
                            [Point(0, 0), Point(1, 1)],
                            [256, 256],
                            CellType.Type.quadrilateral)

V = VectorFunctionSpace(mesh, "DG", 1, dim=4)
u, v = Function(V), TestFunction(V)

mf = MeshFunction("size_t", mesh, 2, 0)
CompiledSubDomain("(x[0] >= 0.5) and (x[1] >= 0.5)").mark(mf, 1)
CompiledSubDomain("(x[0] <= 0.5) and (x[1] >= 0.5)").mark(mf, 2)
CompiledSubDomain("(x[0] <= 0.5) and (x[1] <= 0.5)").mark(mf, 3)
CompiledSubDomain("(x[0] >= 0.5) and (x[1] <= 0.5)").mark(mf, 4)
dx = Measure("dx", subdomain_data=mf)

# Configuration 5
max_t = 0.3
rho0 = [1.0, 2.0, 1.0, 3.0]
p0 = [1.0, 1.0, 1.0, 1.0]
u0 = [-0.75, -0.75, 0.75, 0.75]
v0 = [-0.5, 0.5, 0.5, -0.5]

states = [0, 0, 0, 0]
L = 0
for j in range(4):
    rhoE = energy_density(p0[j], rho0[j], as_vector((u0[j], v0[j])))
    states[j] = Constant((rho0[j], rho0[j]*u0[j], rho0[j]*v0[j], rhoE))
    L += dot(states[j], v)*dx(j+1)

a = dot(TrialFunction(V), v)*dx
A, b = assemble_system(a, L)
solve(A, u.vector(), b)

ceo = CompressibleEulerOperator(mesh, V, DGNeumannBC(ds, u))

PETScOptions.set("ts_monitor")
PETScOptions.set("ts_type", "rosw")
PETScOptions.set("ts_atol", 1e-3)
PETScOptions.set("ts_exact_final_time", "matchstep")

output_vec = Function(V)
def monitor(ts, i, t, x):
    output_vec.vector()[:] = x
    XDMFFile("density.xdmf").write_checkpoint(output_vec.sub(0), "rho", time_step=t, append=i!=0)

class GasDynamics(PETScTSProblem):
    def F(self, u, u_t, u_tt):
        return dot(u_t, v)*dx + ceo.generate_fem_formulation(u, v)

ts = PETScTSSolver(GasDynamics(), u, [])
ts.ts.setMonitor(monitor)
ts.set_from_options()
ts.solve(t0=0.0, dt=1e-4, max_t=max_t, max_steps=200)
