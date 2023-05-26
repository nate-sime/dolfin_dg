import gmsh
import numpy as np
from mpi4py import MPI

import dolfinx


def generate_naca_4digit(comm: MPI.Intracomm, t: float, model_rank: int = 0,
                         n_pts: int = 1000, rounded: bool = False,
                         lc: float = 0.01, m: float | None = None,
                         p: float | None = None,
                         r_farfield: float = 50.0) -> dolfinx.mesh.Mesh:
    """

    Parameters
    ----------
    comm
        MPI communicator
    t
        Chord thickness
    model_rank
        Rank on which to generate mesh
    n_pts
        Number of points in surface discretisation
    rounded
        If true, add rounded trailing edge
    lc
        Characteristic length at surface
    m
        Maximum camber
    p
        Location of maximum camber
    r_farfield
        Far field distance (radius of disk enveloping aerofoil)

    Example
    -------
    NACA0012:
    >>> generate_naca_4digit(comm, t=0.12, p=None, m=None)
    
    NACA2412:
    >>> generate_naca_4digit(comm, t=0.12, p=0.4, m=0.02)

    Returns
        dolfinx mesh
    -------

    """
    gmsh.initialize()

    if comm.rank == model_rank:
        gmsh.clear()

        c = [0.2969, -0.1260, -0.3516, 0.2843]
        c += [-0.1015] if rounded else [-0.1036]

        def y_t(x):
            return 5.0 * t * (c[0] * x ** 0.5 + c[1] * x + c[2] * x ** 2
                              + c[3] * x ** 3 + c[4] * x ** 4)

        if rounded:
            x_r = 0.995
        else:
            x_r = 1.0

        x = np.linspace(0.0, x_r, n_pts)[::-1]
        coords_top = np.vstack((x, y_t(x))).T
        coords_bot = np.vstack((x, -y_t(x))).T

        if m is not None and p is not None:
            def y_c(x):
                return np.where(
                    x <= p,
                    m / p ** 2 * (2 * p * x - x ** 2),
                    m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))

            def dy_c_dx(x):
                return np.where(
                    x <= p,
                    2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))

            theta = np.arctan(dy_c_dx(x))

            x_top = x - y_t(x) * np.sin(theta)
            x_bot = x + y_t(x) * np.sin(theta)

            y_top = y_c(x) + y_t(x) * np.cos(theta)
            y_bot = y_c(x) - y_t(x) * np.cos(theta)

            coords_top = np.vstack((x_top, y_top)).T
            coords_bot = np.vstack((x_bot, y_bot)).T

        pts_top = [gmsh.model.occ.addPoint(*coord, 0, lc) for coord in
                   coords_top]
        pts_bot = [gmsh.model.occ.addPoint(*coord, 0, lc) for coord in
                   coords_bot]

        curv = [gmsh.model.occ.addSpline(pts_top),
                gmsh.model.occ.addSpline(pts_bot)]

        if rounded:
            c = gmsh.model.occ.addPoint(x_r - (1.0 - x_r) * 1e-1, 0.0, 0, lc)
            curv += [gmsh.model.occ.addCircleArc(pts_bot[0], c, pts_top[0])]

        aerofoil = gmsh.model.occ.addCurveLoop(curv)
        aerofoil = gmsh.model.occ.addPlaneSurface([aerofoil])

        disk = gmsh.model.occ.addDisk(0, 0, 0, r_farfield, r_farfield)

        domain, _ = gmsh.model.occ.cut(
            [(2, disk)], [(2, aerofoil)], removeObject=True, removeTool=True)

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [domain[0][1]], 1)

        # TODO: Fix the labelling of facets
        gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, model_rank, gdim=2,
        partitioner=dolfinx.cpp.mesh.create_cell_partitioner(
            dolfinx.mesh.GhostMode.shared_facet))

    gmsh.finalize()

    return mesh
