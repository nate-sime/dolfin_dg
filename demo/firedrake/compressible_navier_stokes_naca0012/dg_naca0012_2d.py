import ufl
import math
import numpy

from firedrake import *
from firedrake.cython import dmplex
from dolfin_dg import *

base = Mesh("naca0012_coarse_mesh.msh")

# Polynomial order
poly_o = 1

# Initial inlet flow conditions
rho_0 = 1.0
M_0 = 0.5
Re_0 = 5e3
p_0 = 1.0
gamma = 1.4
attack = math.radians(2.0)

# Inlet conditions
c_0 = abs(gamma*p_0/rho_0)**0.5
Re = Re_0/(gamma**0.5*M_0)
n_in = numpy.array([cos(attack), sin(attack)])
u_ref = Constant(M_0*c_0)
rho_in = Constant(rho_0)
u_in = u_ref*as_vector((Constant(cos(attack)), Constant(sin(attack))))

# Assign variable names to the inlet, outlet and adiabatic wall BCs. These indices
# will be used to define subsets of the boundary.
INLET = 1
OUTLET = 2
WALL = 3

def mark_mesh(mesh):
    dm = mesh._plex
    sec = dm.getCoordinateSection()
    coords = dm.getCoordinatesLocal()
    dm.removeLabel(dmplex.FACE_SETS_LABEL)
    dm.createLabel(dmplex.FACE_SETS_LABEL)

    faces = dm.getStratumIS("exterior_facets", 1).indices

    for face in faces:
        vertices = dm.vecGetClosure(sec, coords, face).reshape(2, 2)
        midpoint = vertices.mean(axis=0)

        if midpoint[0]**2 + midpoint[1]**2 < 4:
            dm.setLabelValue(dmplex.FACE_SETS_LABEL, face, WALL)
        else:
            if numpy.dot(midpoint, n_in) < 0:
                dm.setLabelValue(dmplex.FACE_SETS_LABEL, face, INLET)
            else:
                dm.setLabelValue(dmplex.FACE_SETS_LABEL, face, OUTLET)
    return mesh

mark_mesh(base)
mh = MeshHierarchy(base, 0)
mesh = mh[-1]

# The initial guess used in the Newton solver. Here we use the inlet flow.
rhoE_in_guess = energy_density(p_0, rho_in, u_in, gamma)
gD_guess = as_vector((rho_in, rho_in*u_in[0], rho_in*u_in[1], rhoE_in_guess))

# Problem function space, (rho, rho*u1, rho*u2, rho*E)
V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
print("Problem size: %d degrees of freedom" % V.dim())

# Use the initial guess.
u_vec = project(gD_guess, V); u_vec.rename("u")
v_vec = TestFunction(V)

# The subsonic inlet, adiabatic wall and subsonic outlet conditions
inflow = subsonic_inflow(rho_in, u_in, u_vec, gamma)
no_slip_bc = no_slip(u_vec)
outflow = subsonic_outflow(p_0, u_vec, gamma)

# Assemble these conditions into DG BCs
dx = dx(degree=4)
dS = dS(degree=4)
ds = ds(degree=4)

bcs = [DGDirichletBC(ds(INLET), inflow),
       DGAdiabticWallBC(ds(WALL), no_slip_bc),
       DGDirichletBC(ds(OUTLET), outflow)]

# Construct the compressible Navier Stokes DG formulation, and compute the symbolic
# Jacobian
ce = CompressibleNavierStokesOperator(mesh, V, bcs, mu=1.0/Re)
F = ce.generate_fem_formulation(u_vec, v_vec, dx=dx, dS=dS)

sp = {
      "snes_type": "newtonls",
      "snes_monitor": None,
      "snes_linesearch_type": "basic",
      "snes_linesearch_maxstep": 1,
      "snes_linesearch_damping": 1,
      "snes_linesearch_monitor": None,
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"
     }
solve(F == 0, u_vec, solver_parameters=sp)

# Assemble variables required for the lift and drag computation
n = FacetNormal(mesh)
rho, u, E = flow_variables(u_vec)
Q = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "DG", 1)
rho = project(rho, Q)
u = project(u, W)
E = project(E, Q)
rho.rename("Density"); u.rename("Velocity"); E.rename("Energy")
p = pressure(u_vec, gamma)
l_ref = Constant(1.0)
tau = 1.0/Re*(grad(u) + grad(u).T - 2.0/3.0*(tr(grad(u)))*Identity(2))
C_infty = 0.5*rho_in*u_ref**2*l_ref

# Assemble the homogeneity tensor with dot(grad(T), n) = 0 for use in the
# adjoint consistent lift and drag formulation
h = CellVolume(mesh)/FacetArea(mesh)
sigma = Constant(20.0)*Constant(max(poly_o**2, 1))/h
G_adiabitic = homogeneity_tensor(ce.F_v_adiabatic, u_vec)
G_adiabitic = ufl.replace(G_adiabitic, {u_vec: no_slip_bc})

# The drag coefficient
psi_drag = as_vector((cos(attack), sin(attack)))
drag = 1.0/C_infty*dot(psi_drag, p*n - tau*n)*ds(WALL)

# The adjoint consistent drag coefficient
z_drag = 1.0/C_infty*as_vector((0, psi_drag[0], psi_drag[1], 0))
drag += inner(sigma*hyper_tensor_product(G_adiabitic, dg_outer(u_vec - no_slip_bc, n)), dg_outer(z_drag, n))*ds(WALL)

# The lift coefficient
psi_lift = as_vector((-sin(attack), cos(attack)))
lift = 1.0/C_infty*dot(psi_lift, p*n - tau*n)*ds(WALL)

result = (V.dim(), assemble(drag), assemble(lift))
print("DoFs: %d, Drag: %.5e, Lift: %.5e" % result)

File("output/solution.pvd").write(u, rho, E)
