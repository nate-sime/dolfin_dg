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
    vn = len(v)
    L = [None for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        L[vi] = fs.split(F, v[vi])
        L[vi] = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(L[vi])
        L[vi] = ufl.algorithms.apply_derivatives.apply_derivatives(L[vi])

    return L


def extract_blocks(F, u, v):
    un, vn = len(u), len(v)
    a = [[None for _ in range(un)] for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        for ui in range(un):
            a[vi][ui] = fs.split(F, v[vi], u[ui])
            a[vi][ui] = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(a[vi][ui])
            a[vi][ui] = ufl.algorithms.apply_derivatives.apply_derivatives(a[vi][ui])

    return a


def extract_block_linear_system(F, u, v):
    F_a = extract_blocks(F, u, v)
    F_L = extract_rows(F, v)

    a = list(list(map(ufl.lhs, row)) for row in F_a)
    L = list(map(ufl.rhs, F_L))

    return a, L


def derivative_block(F, u, du=None, coefficient_derivatives=None):
    import ufl
    if isinstance(F, ufl.Form):
        return ufl.derivative(F, u, du, coefficient_derivatives)

    if not isinstance(F, (list, tuple)):
        raise TypeError("Expecting F to be a list of Forms. Found: %s" % str(F))

    if not isinstance(u, (list, tuple)):
        raise TypeError("Expecting u to be a list of Coefficients. Found: %s" % str(u))

    if du is not None:
        if not isinstance(du, (list, tuple)):
            raise TypeError("Expecting du to be a list of Arguments. Found: %s" % str(u))

    import itertools
    from ufl.algorithms.apply_derivatives import apply_derivatives
    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering

    m, n = len(u), len(F)

    if du is None:
        du = [None] * m

    J = [[None for _ in range(m)] for _ in range(n)]

    for (i, j) in itertools.product(range(n), range(m)):
        gateaux_derivative = ufl.derivative(F[i], u[j], du[j], coefficient_derivatives)
        gateaux_derivative = apply_derivatives(apply_algebra_lowering(gateaux_derivative))
        if gateaux_derivative.empty():
            gateaux_derivative = None
        J[i][j] = gateaux_derivative

    return J
