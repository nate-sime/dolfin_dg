import ufl


class SeparateSpaceFormSplitter(ufl.corealg.multifunction.MultiFunction):

    def split(self, form, v, u=None):
        self.vu = tuple((v, u))
        return ufl.algorithms.map_integrands.map_integrand_dags(self, form)

    def argument(self, obj):
        if obj not in self.vu:
            return ufl.constantvalue.Zero(shape=obj.ufl_shape)
        return obj

    expr = ufl.corealg.multifunction.MultiFunction.reuse_if_untouched


def extract_rows(F, v):
    """
    Parameters
    ----------
    F
        UFL residual formulation
    v
        Ordered list of test functions

    Returns
    -------
    List of extracted block residuals ordered corresponding to each test
    function
    """
    vn = len(v)
    L = [None for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        # Do the initial split replacing testfunctions with zero
        L[vi] = fs.split(F, v[vi])

        # Now remove the empty forms. Why don't FFC/UFL do this already?
        L_reconstruct = []
        for integral in L[vi].integrals():
            arguments = ufl.algorithms.analysis.extract_arguments(integral)
            if len(arguments) < 1:
                continue

            # Sanity checks: Should be only one test function, and it should
            # be the one we want to keep
            assert len(arguments) == 1
            assert arguments[0] is v[vi]
            L_reconstruct.append(integral)

        # Generate the new form with the removed zeroes
        L[vi] = ufl.Form(L_reconstruct)
    return L


def extract_blocks(F, u, v):
    """
    Parameters
    ----------
    F
        UFL residual formulation
    u
        Ordered list of trial functions
    v
        Ordered list of test functions

    Returns
    -------
    A nested list of block matrix components ordered by test function rows
    and trial function columns
    """
    un, vn = len(u), len(v)
    a = [[None for _ in range(un)] for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        for ui in range(un):
            a[vi][ui] = fs.split(F, v[vi], u[ui])
            a[vi][ui] = ufl.algorithms.apply_algebra_lowering.\
                apply_algebra_lowering(a[vi][ui])
            a[vi][ui] = ufl.algorithms.apply_derivatives.\
                apply_derivatives(a[vi][ui])

    return a


def extract_block_linear_system(F, u, v):
    """
    Parameters
    ----------
    F
        UFL residual formulation
    u
        Ordered list of trial functions
    v
        Ordered list of test functions

    Returns
    -------
    The `left hand side' and `right hand side' block matrix and vector
    formulation, respectively
    """
    F_a = extract_blocks(F, u, v)
    F_L = extract_rows(F, v)

    a = list(list(map(ufl.lhs, row)) for row in F_a)
    L = list(map(ufl.rhs, F_L))

    return a, L


def derivative_block(F, u, du=None, coefficient_derivatives=None):
    """
    Parameters
    ----------
    F
        Block residual formulation
    u
        Ordered solution functions
    du
        Ordered trial functions
    coefficient_derivatives
        Prescribed derivative map

    Returns
    -------
    Block matrix corresponding to the ordered components of the
    Gateaux/Frechet derivative.
    """
    import ufl
    if isinstance(F, ufl.Form):
        return ufl.derivative(F, u, du, coefficient_derivatives)

    if not isinstance(F, (list, tuple)):
        raise TypeError("Expecting F to be a list of Forms. Found: %s"
                        % str(F))

    if not isinstance(u, (list, tuple)):
        raise TypeError("Expecting u to be a list of Coefficients. Found: %s"
                        % str(u))

    if du is not None:
        if not isinstance(du, (list, tuple)):
            raise TypeError("Expecting du to be a list of Arguments. Found: %s"
                            % str(u))

    import itertools
    from ufl.algorithms.apply_derivatives import apply_derivatives
    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering

    m, n = len(u), len(F)

    if du is None:
        du = [None] * m

    J = [[None for _ in range(m)] for _ in range(n)]

    for (i, j) in itertools.product(range(n), range(m)):
        gateaux_derivative = ufl.derivative(F[i], u[j], du[j],
                                            coefficient_derivatives)
        gateaux_derivative = apply_derivatives(
            apply_algebra_lowering(gateaux_derivative))
        if gateaux_derivative.empty():
            gateaux_derivative = None
        J[i][j] = gateaux_derivative

    return J
