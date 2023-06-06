import gmsh
import numpy as np
import scipy.integrate
from mpi4py import MPI

import dolfinx


def parse_naca_digits(digits: str):
    """
    Given the NACA designation digits, return parameters of camber, location
    and chord thickness.

    Parameters
    ----------
    digits
        String representation of digits

    Returns
    -------
    NACA parameters

    Examples
    --------
    NACA0012:
    >>> parse_naca_digits("0012")

    NACA2412:
    >>> parse_naca_digits("2412")
    """
    digits = str(digits)
    if len(digits) == 4:
        m = float(digits[0]) / 100.0
        p = float(digits[1]) / 10.0
        t = float(digits[2:]) / 100.0
        return m, p, t


def generate_naca_4digit(comm: MPI.Intracomm, m: float, p: float, t: float,
                         model_rank: int = 0, rounded: bool = False,
                         lc: float = 0.01, r_farfield: float = 50.0,
                         gmsh_options: dict[str, int] = None
                         ) -> dolfinx.mesh.Mesh:
    """

    Parameters
    ----------
    comm
        MPI communicator
    m
        Maximum camber
    p
        Location of maximum camber
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
    r_farfield
        Far field distance (radius of disk enveloping aerofoil)

    Returns
    -------
        dolfinx mesh

    Example
    -------
    NACA0012:
    >>> generate_naca_4digit(comm, 0.0, 0.0, 0.12)

    NACA2412:
    >>> generate_naca_4digit(comm, 0.02, 0.4, 0.12)


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

        def create_edge_function(start, end, TOL=1e-12):
            dx = end[0] - start[0]
            if abs(dx) < TOL:
                c = end[0]
                f = lambda x: x - c
            else:
                m = (end[1] - start[1])/(end[0] - start[0])
                c = end[1] - m*end[0]
                f = lambda x: m*x + c
            return f

        def compute_edge_error_h1(x_top, f_aerofoil, fprime_airfoil):
            err_a = []
            for j in range(len(x_top)-1):
                a, b = x_top[j:j+2]
                edge_f = create_edge_function([a, f_aerofoil(a)], [b, f_aerofoil(b)])
                gradient = (f_aerofoil(b) - f_aerofoil(a)) / (b - a)
                error_func = lambda x: (f_aerofoil(x) - edge_f(x))**2 + (fprime_airfoil(x) - gradient)**2
                err = scipy.integrate.quad(error_func, a, b)[0]**0.5
                err_a.append(err)
            return np.array(err_a, dtype=np.double)


        def generate_x_points_h1(f_aerofoil, x0, eps=1e-3, max_it=20):
            import sympy
            xsym = sympy.Symbol("x", real=True)
            fprime_airfoil = sympy.lambdify(xsym, f_aerofoil(xsym).diff(xsym))
            x = np.array(x0, dtype=np.double)
            for j in range(max_it):
                err = compute_edge_error_h1(x, f_aerofoil, fprime_airfoil)
                err_condition = err < eps
                if np.all(err_condition):
                    break
                new_idxs = np.where(~err_condition)[0]
                x = np.insert(x, new_idxs+1, (x[new_idxs] + x[new_idxs+1])/2.0)
            return x

        x = generate_x_points_h1(y_t, [1e-8, x_r])
        x = np.insert(x, [0], 0.0)

        if abs(m) > 1e-12 and abs(p) > 1e-12:
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
        else:
            x_top, x_bot = x, x
            y_top, y_bot = y_t(x), -y_t(x)

        coords_top = np.vstack((x_top, y_top)).T
        coords_bot = np.vstack((x_bot, y_bot)).T

        # import matplotlib.pyplot as plt
        # plt.plot(coords_bot[:,0], coords_bot[:,1], "-x")
        # plt.plot(coords_top[:,0], coords_top[:,1], "-x")
        # plt.gca().axis("equal")
        # plt.grid()
        # plt.show()
        # quit()

        pts_top = [gmsh.model.occ.addPoint(*coord, 0, lc) for coord in
                   coords_top]
        pts_bot = [gmsh.model.occ.addPoint(*coord, 0, lc) for coord in
                   coords_bot]

        curv = [gmsh.model.occ.addSpline(pts_top),
                gmsh.model.occ.addSpline(pts_bot)]

        if rounded:
            c = gmsh.model.occ.addPoint(x_r - (1.0 - x_r) * 1e-1, 0.0, 0, lc)
            curv += [gmsh.model.occ.addCircleArc(pts_bot[-1], c, pts_top[-1])]

        aerofoil = gmsh.model.occ.addCurveLoop(curv)
        aerofoil = gmsh.model.occ.addPlaneSurface([aerofoil])

        disk = gmsh.model.occ.addDisk(0, 0, 0, r_farfield, r_farfield)

        domain, _ = gmsh.model.occ.cut(
            [(2, disk)], [(2, aerofoil)], removeObject=True, removeTool=True)

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [domain[0][1]], 1)

        # TODO: Fix the labelling of facets
        if gmsh_options:
            for k, v in gmsh_options.items():
                gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, model_rank, gdim=2,
        partitioner=dolfinx.cpp.mesh.create_cell_partitioner(
            dolfinx.mesh.GhostMode.shared_facet))

    gmsh.finalize()

    return mesh
