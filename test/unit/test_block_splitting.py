from dolfin import (
    Function, FunctionSpace, VectorElement, FiniteElement, Identity, grad,
    div, inner, dx, as_backend_type, assemble, TrialFunction, FacetNormal,
    Constant, Expression, UnitSquareMesh, TestFunction, sym, jump, dS,
    derivative, assemble_system)

import dolfin_dg


def test_block_split():
    mesh = UnitSquareMesh(4, 4)

    Ve = VectorElement("CG", mesh.ufl_cell(), 1)
    Qe = FiniteElement("CG", mesh.ufl_cell(), 1)

    # Block
    V = FunctionSpace(mesh, Ve)
    Q = FunctionSpace(mesh, Qe)

    u, v = Function(V), TestFunction(V)
    p, q = Function(Q), TestFunction(Q)

    u.interpolate(Expression(("x[0]", "x[1]"), degree=1))
    p.interpolate(Expression("x[0]*x[0] + x[1]*x[1]", degree=1))

    mu = Constant(1.0)

    F0 = inner(2 * mu * sym(grad(u)) - p * Identity(2), grad(v)) * dx
    F1 = inner(q, div(u)) * dx
    F = sum((F0, F1))

    # Test rows
    rows = dolfin_dg.extract_rows(F, [v, q])

    F0b = as_backend_type(assemble(F0))
    F1b = as_backend_type(assemble(F1))

    b1 = as_backend_type(assemble(rows[0]))
    b2 = as_backend_type(assemble(rows[1]))

    assert F0b.vec().norm() == b1.vec().norm()
    assert F1b.vec().norm() == b2.vec().norm()

    # Test matrices
    n = FacetNormal(mesh)
    u, p = TrialFunction(V), TrialFunction(Q)

    a00 = inner(2 * mu * sym(grad(u)), grad(v)) * dx \
        + inner(jump(u), jump(v)) * dS
    a01 = -inner(p * Identity(2), grad(v)) * dx \
        + inner(jump(p, n), jump(v)) * dS
    a10 = inner(q, div(u)) * dx + inner(u("+"), q("-") * n("-")) * dS

    a = sum((a00, a01, a10))

    blocks = dolfin_dg.extract_blocks(a, [u, p], [v, q])

    A00 = as_backend_type(assemble(a00))
    A00_a = as_backend_type(assemble(blocks[0][0]))
    assert A00.mat().norm() == A00_a.mat().norm()

    A01 = as_backend_type(assemble(a01))
    A01_a = as_backend_type(assemble(blocks[0][1]))
    assert A01.mat().norm() == A01_a.mat().norm()

    A10 = as_backend_type(assemble(a10))
    A10_a = as_backend_type(assemble(blocks[1][0]))
    assert A10.mat().norm() == A10_a.mat().norm()


def test_block_split_dg():
    mesh = UnitSquareMesh(4, 4)

    Ve = VectorElement("DG", mesh.ufl_cell(), 1)
    Qe = FiniteElement("DG", mesh.ufl_cell(), 1)

    # Block
    V = FunctionSpace(mesh, Ve)
    Q = FunctionSpace(mesh, Qe)

    u, v = Function(V), TestFunction(V)
    p, q = Function(Q), TestFunction(Q)

    u.interpolate(Expression(("x[0]", "x[1]"), degree=1))
    p.interpolate(Expression("x[0]*x[0] + x[1]*x[1]", degree=1))

    def F_v(u, grad_u, p_local=None):
        if p_local is None:
            p_local = p
        return 2 * sym(grad_u) - p_local * Identity(mesh.geometry().dim())

    stokes = dolfin_dg.StokesOperator(None, None, [], None)

    # Residual, Jacobian and preconditioner FE formulations
    F = stokes.generate_fem_formulation(
        u, v, p, q, eta=lambda x: 1)
    F0, F1 = dolfin_dg.block.extract_rows(F, [v, q])

    F0 += inner(F_v(u, grad(u), p), grad(v)) * dx
    F1 += inner(q, div(u)) * dx

    F = sum((F0, F1))

    # Test rows
    rows = dolfin_dg.extract_rows(F, [v, q])

    F0b = as_backend_type(assemble(F0))
    F1b = as_backend_type(assemble(F1))

    b0 = as_backend_type(assemble(rows[0]))
    b1 = as_backend_type(assemble(rows[1]))

    assert F0b.vec().norm() == b0.vec().norm()
    assert F1b.vec().norm() == b1.vec().norm()

    # Test matrices
    J00 = derivative(F0, u)
    J01 = derivative(F0, p)
    J10 = derivative(F1, u)

    dU = list(map(TrialFunction, (V, Q)))
    J = dolfin_dg.derivative_block(rows, [u, p], dU)

    # Test regular assembler
    A00 = as_backend_type(assemble(J00))
    A00_a = as_backend_type(assemble(J[0][0]))
    assert A00.mat().norm() == A00_a.mat().norm()

    A01 = as_backend_type(assemble(J01))
    A01_a = as_backend_type(assemble(J[0][1]))
    assert A01.mat().norm() == A01_a.mat().norm()

    A10 = as_backend_type(assemble(J10))
    A10_a = as_backend_type(assemble(J[1][0]))
    assert A10.mat().norm() == A10_a.mat().norm()

    # Test system assembler
    A00, F0b = map(as_backend_type, assemble_system(J00, F0))
    A00_a, b0 = map(as_backend_type, assemble_system(J[0][0], rows[0]))
    assert F0b.vec().norm() == b0.vec().norm()
    assert A00.mat().norm() == A00_a.mat().norm()

    A01, F0b = map(as_backend_type, assemble_system(J01, F0))
    A01_a, b0 = map(as_backend_type, assemble_system(J[0][1], rows[0]))
    assert F0b.vec().norm() == b0.vec().norm()
    assert A01.mat().norm() == A01_a.mat().norm()

    A10, F1b = map(as_backend_type, assemble_system(J10, F1))
    A10_a, b1 = map(as_backend_type, assemble_system(J[1][0], rows[1]))
    assert F1b.vec().norm() == b1.vec().norm()
    assert A10.mat().norm() == A10_a.mat().norm()

    # TODO: There's something different between Assembler/SystemAssembler
    #  even when using empty BCs. Need to test this here.
