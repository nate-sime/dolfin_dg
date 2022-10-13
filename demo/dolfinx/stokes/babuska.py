import gmsh
import ufl
from mpi4py import MPI
import dolfinx.io.gmshio
import ufl
import dolfin_dg
import dolfin_dg.dolfinx
import numpy as np


def generate_mesh(cell_size: float, shape_eps: float, p_mesh: int):
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh_shape = gmsh.model.occ.addDisk(0, 0, 0, shape_eps, 1)
        point = gmsh.model.occ.addPoint(0, 0, 0)
        gmsh.model.occ.fragment([(0, point)], [(2, gmsh_shape)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [gmsh_shape])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cell_size)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(p_mesh)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
    gmsh.finalize()
    return mesh


cell_sizes = np.array([2**-n for n in range(2, 7)], dtype=np.double)
ul2_errors = np.zeros_like(cell_sizes)
shape_eps = 1.0
p_mesh = 1
p_order = 2

for j, cell_size in enumerate(cell_sizes):
    mesh = generate_mesh(cell_size, shape_eps, p_mesh)

    We = ufl.MixedElement([
        ufl.VectorElement("CG", mesh.ufl_cell(), degree=p_order),
        ufl.FiniteElement("CG", mesh.ufl_cell(), degree=p_order - 1)])
    W = dolfinx.fem.FunctionSpace(mesh, We)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(mesh)

    # -- Zero slip on (-1, 1)^2 rectangle
    u_soln = ufl.as_vector((2*x[1]*(1.0 - x[0]*x[0]),
                            -2*x[0]*(1.0 - x[1]*x[1])))

    # -- Exact representation no slip circle
    # u_soln = ufl.as_vector((-x[1], x[0]))

    # -- Inexact representation no slip circle
    # u_soln = ufl.as_vector((-x[1]*ufl.sqrt(ufl.Max(x[0]**2 + x[1]**2, 0.0)),
    #                         x[0]*ufl.sqrt(ufl.Max(x[0]**2 + x[1]**2, 0.0))))

    f = -ufl.div(2*ufl.sym(ufl.grad(u_soln)))

    def F_v(u, grad_u):
        return 2*ufl.sym(grad_u) - p * ufl.Identity(mesh.geometry.dim)

    a = (
                ufl.inner(F_v(u, ufl.grad(u)), ufl.grad(v))
                - ufl.inner(ufl.div(u), q)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    n = ufl.FacetNormal(mesh)
    g_tau = dolfin_dg.tangential_proj(2*ufl.sym(ufl.grad(u_soln)) * n, n)

    hc = dolfin_dg.dolfinx.cell_volume_dg0(mesh)
    hf = dolfin_dg.dolfinx.facet_area_avg_dg0(mesh)
    h = hc/hf
    sigma = dolfinx.fem.Constant(mesh, 20.0 * p_order ** 2) / h

    nitsche_bc = dolfin_dg.nitsche.StokesNitscheBoundary(F_v, u, p, v, q, C_IP=sigma)
    F_slip = nitsche_bc.slip_nitsche_bc_residual(
        u_soln, g_tau, ufl.ds)

    a += ufl.lhs(F_slip)
    L += ufl.rhs(F_slip)

    Q = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order - 1))
    zero = dolfinx.fem.Function(Q)
    zero.x.set(0.0)
    dofs = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
    bc2 = dolfinx.fem.dirichletbc(zero, dofs, W.sub(1))

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    l2error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((uh.sub(0) - u_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)
    ul2_errors[j] = l2error_u

    if mesh.comm.rank == 0:
        print(f"run {j}: cell_size={cell_size}, u L2 error={l2error_u}")

hrates = np.log(cell_sizes[:-1] / cell_sizes[1:])
rates_u = np.log(ul2_errors[:-1] / ul2_errors[1:]) / hrates
if mesh.comm.rank == 0:
    print(f"u L2 convergence rates: {rates_u}")
    import matplotlib.pyplot as plt
    plt.loglog(cell_sizes, ul2_errors, "-o")
    plt.grid()
    plt.title(f"Ellipse $(a, b) = ({shape_eps}, 1.0)$, "
              f"mesh degree ${p_mesh}$, "
              f"$p = {p_order}$")
    plt.xlabel("$h$")
    plt.ylabel("$\Vert u - u_\\mathrm{soln} \Vert_{L_2(\\Omega)}$")
    plt.show()
