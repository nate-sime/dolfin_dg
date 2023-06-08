import gmsh
import numpy as np
import scipy.integrate
from mpi4py import MPI

import dolfinx


def _create_edge_function(start, end, TOL=1e-12):
    """
    Generate a callable function
        f(x) = m x + c
    which represents the linear element between the start and end points. A
    tolerance is provided for lines parallel with the y-axis.
    """
    dx = end[0] - start[0]
    if abs(dx) < TOL:
        c = end[0]
        f = lambda x: x - c
    else:
        m = (end[1] - start[1]) / (end[0] - start[0])
        c = end[1] - m * end[0]
        f = lambda x: m * x + c
    return f


def _compute_edge_error_h1(x, f_aerofoil, fprime_aerofoil):
    """
    Given a function f_aerofoil and its derivative fprime_aerofoil,
    compute the H^1 error of the piecewise linear interpolation at points
    x.
    """
    err_a = []
    for j in range(len(x) - 1):
        a, b = x[j:j + 2]
        edge_f = _create_edge_function([a, f_aerofoil(a)], [b, f_aerofoil(b)])
        gradient = (f_aerofoil(b) - f_aerofoil(a)) / (b - a)
        error_func = lambda x: (f_aerofoil(x) - edge_f(x)) ** 2 + (
                    fprime_aerofoil(x) - gradient) ** 2
        err = scipy.integrate.quad(error_func, a, b)[0] ** 0.5
        err_a.append(err)
    return np.array(err_a, dtype=np.double)


def _generate_x_points_h1(f_aerofoil, x0, eps=1e-3, max_it=20):
    """
    Given a function f_aerofoil, compute the points which minimise its
    piecewise linear interpolation error as measured in the H^1 norm using
    bisection refinement.
    """
    import sympy
    xsym = sympy.Symbol("x", real=True)
    fprime_aerofoil = sympy.lambdify(xsym, f_aerofoil(xsym).diff(xsym))
    x = np.array(x0, dtype=np.double)
    for j in range(max_it):
        err = _compute_edge_error_h1(x, f_aerofoil, fprime_aerofoil)
        err_condition = err < eps
        if np.all(err_condition):
            break
        new_idxs = np.where(~err_condition)[0]
        x = np.insert(x, new_idxs + 1, (x[new_idxs] + x[new_idxs + 1]) / 2.0)
    return x


def y_t(xi, t, rounded):
    """
    Symmetric part of NACA00xx aerofoil

    Parameters
    ----------
    xi
        Reference coordinates
    t
        Chord thickness
    rounded
        If true, add rounded trailing edge

    Returns
    -------
        Top part of aerofoil y-coordinates
    """
    c = [0.2969, -0.1260, -0.3516, 0.2843,
         -0.1015 if rounded else -0.1036]
    y = 5.0 * t * (c[0] * xi ** 0.5 + c[1] * xi + c[2] * xi ** 2
                   + c[3] * xi ** 3 + c[4] * xi ** 4)
    return y


def naca_4digit_coordinates(xi: np.ndarray, m: float, p: float, t: float,
                            rounded: bool) -> tuple[np.ndarray]:
    """
    Given reference coordinates, compute on the bottom and top
    coordinates of a 4digit NACA aerofoil

    Parameters
    ----------
    xi
        Reference coordinates
    m
        Maximum camber
    p
        Location of maximum camber
    t
        Chord thickness
    rounded
        If true, add rounded trailing edge

    Returns
    -------
        list of bottom and top coordinate sets
    """
    # Symmetric part
    y_sym = lambda x: y_t(x, t, rounded)

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

        theta = np.arctan(dy_c_dx(xi))

        x_top = xi - y_sym(xi) * np.sin(theta)
        x_bot = xi + y_sym(xi) * np.sin(theta)

        y_top = y_c(xi) + y_sym(xi) * np.cos(theta)
        y_bot = y_c(xi) - y_sym(xi) * np.cos(theta)
    else:
        x_top, x_bot = xi, xi
        y_top, y_bot = y_sym(xi), -y_sym(xi)

    coords_top = np.vstack((x_top, y_top)).T
    coords_bot = np.vstack((x_bot, y_bot)).T

    return coords_bot, coords_top


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
    Generate a mesh of NACA 4 digit aerofoil.

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
        Far field distance (distance from enveloping rectangle to aerofoil)

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

        if rounded:
            x_r = 0.995
        else:
            x_r = 1.0

        y_sym = lambda xi: y_t(xi, t, rounded)
        xi = _generate_x_points_h1(y_sym, [1e-8, x_r])
        xi = np.insert(xi, [0], 0.0)

        coords_top, coords_bot = naca_4digit_coordinates(xi, m, p, t, rounded)

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

        # Using disc interferes with gmsh's refinement by curvature
        # disk = gmsh.model.occ.addDisk(0, 0, 0, r_farfield, r_farfield)
        disk = gmsh.model.occ.addRectangle(
            -r_farfield, -r_farfield, 0.0, 2*r_farfield, 2*r_farfield)

        domain, _ = gmsh.model.occ.cut(
            [(2, disk)], [(2, aerofoil)], removeObject=True, removeTool=True)

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [domain[0][1]], 1)

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
