import pytest
import numpy
import ufl
from ufl.algorithms.apply_derivatives import apply_derivatives

import dolfin_dg
import dolfin_dg.dg_ufl as dg
from dolfin import *


mesh = UnitSquareMesh(3, 3)

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)
dS = Measure("dS", domain=mesh)

p = Function(FunctionSpace(mesh, "DG", 1))
p.vector()[:] = numpy.arange(p.vector()[:].shape[0])
a = Function(FunctionSpace(mesh, "DG", 1))
a.vector()[:] = numpy.arange(a.vector()[:].shape[0])
b = Function(FunctionSpace(mesh, "DG", 1))
b.vector()[:] = numpy.arange(b.vector()[:].shape[0])

V = VectorFunctionSpace(mesh, "DG", 1)
u = Function(V)
v = TestFunction(V)
u.vector()[:] = numpy.arange(u.vector()[:].shape[0])

n = FacetNormal(mesh)


scalar_comparisons = (
    # Standard avg rules
    (avg(p) * dS, dg.avg(p) * dS),
    (avg(p) * dS, dg.avg(dg.avg(p)) * dS),
    (avg(p) * dS, dg.avg(dg.avg(dg.avg(p))) * dS),
    # ---
    # Standard jump rules
    (jump(p) * dS, dg.jump(p) * dS),
    (Constant(0.0) * dS, (Constant(0.0) + dg.avg(dg.jump(p))) * dS),
    (2*jump(p) * dS, dg.jump(dg.jump(p)) * dS),
    (4*jump(p) * dS, dg.jump(dg.jump(dg.jump(p))) * dS),
    ((3*jump(p)) * dS, dg.jump(p + dg.jump(p)) * dS),
    # ---
    # Commutitive
    (jump(2 * p) * dS, dg.jump(2 * p) * dS),
    (jump(2 * p) * jump(4 * p) * dS, dg.jump(2 * p) * dg.jump(4 * p) * dS),
    (avg(2 * p) * dS, dg.avg(2 * p) * dS),
    (avg(2 * p) * jump(4 * p) * dS, dg.avg(2 * p) * dg.jump(4 * p) * dS),
    # ---
    # Distributive
    (jump(2 * p + p) * dS, dg.jump(2 * p + p) * dS),
    (avg(2 * p + p) * dS, dg.avg(2 * p + p) * dS),
    (jump(2 * (p + p)) * dS, dg.jump(2 * (p + p)) * dS),
    (avg(2 * (p + p)) * dS, dg.avg(2 * (p + p)) * dS),
    (avg(p * (p + 1)) * dS, dg.avg(p * (p + 1)) * dS),
    (avg(p ** (p + 1)) * dS, dg.avg(p ** (p + 1)) * dS),
    # ---
    # Statements which may be simplified
    # ---
    # avg(a + avg(b)) = avg(a + b) = avg(a) + avg(b)
    ((avg(a) + avg(b)) * dS, dg.avg(a + dg.avg(b)) * dS),
    ((avg(a + b) + avg(1 + a + (a + b)**2)) * dS, dg.avg(a + b + dg.avg(1 + a + (a+b)**2)) * dS),
    ((2*avg(a + b ) + avg(2*(a + b)) + avg((a + b)**2)) * dS, dg.avg(dg.avg(2*(a + b)) + 2*(a + b) + dg.avg((a+b)**2)) * dS),
    # ---
    # jump(a + avg(b)) = jump(a)
    (jump(a) * dS, dg.jump(a + dg.avg(b)) * dS),
    (jump(a + b) * dS, dg.jump(a + b + dg.avg(b + a)) * dS),
    (jump((a + b)**2) * dS, dg.jump((a + b)**2 + dg.avg(b + a)**2) * dS),
    # ---
    # avg(avg(a) * avg(b)) = avg(a) * avg(b)
    (avg(a)*avg(b) * dS, dg.avg(dg.avg(a)*dg.avg(b)) * dS),
    (avg(a + b)*avg(b) * dS, dg.avg(dg.avg(a + b)*dg.avg(b)) * dS),
    (avg(a + b)*avg(b**2) * dS, dg.avg(dg.avg(a + b)*dg.avg(b**2)) * dS),
    ((avg(2*b) + avg(a + b)*avg(b**2)) * dS, dg.avg(2 * b + dg.avg(a + b)*dg.avg(b**2)) * dS),
    # ---
    # avg(a * avg(b)) = avg(a) * avg(b)
    (avg(a) * avg(b) * dS, dg.avg(a * dg.avg(b)) * dS),
    # ---
    # avg(a * jump(b)) =  avg(a) * jump(b)
    (Constant(0.0) * dS, (Constant(0.0) + dg.avg(a * dg.jump(b))) * dS),
    # # (avg(a * a) * jump(b) * dS, dg.avg(a * dg.jump(b) * a) * dS), # TODO
    # # (avg(a) * jump(b)**2 * dS, dg.avg(a * dg.jump(b)**2) * dS),   # TODO
    (avg(a*b) * dS, dg.avg(a*b + a * dg.jump(b)) * dS),
    (avg(a) * dS, dg.avg(a + a*dg.jump(b*dg.avg(a))) * dS),
    # ---
    # avg(jump(a) * avg(b)) = jump(a) * jump(b)
    (Constant(0.0) * dS, (Constant(0.0) + dg.avg(dg.jump(a) * dg.avg(b))) * dS),
    (Constant(0.0) * dS, (Constant(0.0) + dg.avg(dg.jump(a + b) * dg.avg(b**2 + 2 * b))) * dS),
    # ---
    # 2*jump(a, n) = jump(avg(a) + jump(a), n)
    ((2 * jump(a, n))**2 * dS, dg.jump(dg.avg(a) + dg.jump(a), n)**2 * dS),
    # ---
    # 2 * jump(a) * avg(b) = jump(jump(a)*avg(b))
    # (2*jump(a)*avg(b) * dS, (Constant(0.0) + dg.jump(dg.jump(a)*dg.avg(b))) * dS), # TODO
)

@pytest.mark.parametrize("comparison", scalar_comparisons)
def test_scalar(comparison):
    assert assemble(comparison[0]) == pytest.approx(assemble(dg.apply_dg_operator_lowering(comparison[1])), rel=1e-15)


scalar_facet_normal_comparisons = (
    # Standard jump rules
    (jump(p, n)**2*dS, dg.jump(p, n)**2*dS),
    (jump(p, n)**2*dS, dg.avg(dg.jump(p, n)**2)*dS),
    (jump(p, n)**2*dS, dg.avg(dg.avg(dg.jump(p, n)**2))*dS),
    # Commutitive
    (jump(2*p, n)**2*dS, dg.jump(2*p, n)**2*dS),
    (dot(jump(2*p, n), jump(4*p, n))*dS, dot(dg.jump(2*p), dg.jump(4*p))*dS),
    (avg(2*p)*jump(4*p, n)**2*dS, dg.avg(2*p)*dg.jump(4*p, n)**2*dS),
    # Distributive
    (jump(2*p + p, n)**2*dS, dg.jump(2*p + p, n)**2*dS),
    (jump(2*p, n)**2*dS, dg.jump(2*p + dg.avg(p), n)**2*dS),
    # ---
    # Statements which may be simplified
    # jump(a + avg(b)) = jump(a)
    (jump(a, n)**2 * dS, dg.jump(a + dg.avg(b), n)**2 * dS),
    (jump(a + b, n)**2 * dS, dg.jump(a + b + dg.avg(b + a), n)**2 * dS),
    (jump((a + b)**2, n)**2 * dS, dg.jump((a + b)**2 + dg.avg(b + a)**2, n)**2 * dS),
    # ---
    # jump(jump(a), n) = 2*jump(a, n)
    ((2*jump(a, n))**2 * dS, dg.jump(dg.jump(a), n)**2 * dS),
    ((6*jump(a, n))**2 * dS, dg.jump(dg.jump(a) + 2*dg.jump(a), n)**2 * dS),
    # ---
    # 2*jump(a, n) = jump(avg(a) + jump(a), n)
    (2*jump(u, n) * dS, dg.jump(dg.avg(u) + dg.jump(u), n) * dS),
)


@pytest.mark.parametrize("comparison", scalar_facet_normal_comparisons)
def test_scalar_with_facet_normal(comparison):
    assert assemble(comparison[0]) == pytest.approx(assemble(dg.apply_dg_operator_lowering(comparison[1])), rel=1e-15)


vector_facet_normal_comparisons = (
    # Standard jump rules
    (jump(u, n)**2*dS, dg.jump(u, n)**2*dS),
    (jump(u, n)**2*dS, dg.avg(dg.jump(u, n)**2)*dS),
    (jump(u, n)**2*dS, dg.avg(dg.avg(dg.jump(u, n)**2))*dS),
    # Commutitive
    (jump(2*u, n)**2*dS, dg.jump(2*u, n)**2*dS),
    (dot(jump(2*u, n), jump(4*u, n))*dS, dot(dg.jump(2*u, n), dg.jump(4*u, n))*dS),
    (dot(avg(2*u), jump(4*u))*dS, dot(dg.avg(2*u), dg.jump(4*u))*dS),
    (dot(avg(2*u), jump(u))*dS, dot(dg.avg(2*u), dg.jump(u))*dS),
    # Distributive
    (jump(2*u + u)**2*dS, dg.jump(2*u + u)**2*dS),
    (jump(2*u, n)**2*dS, dg.jump(2*u + dg.avg(u), n)**2*dS),
    # ---
    # statements for simplification
    # jump(jump(u), n) = 2 * jump(u, n)
    (2*jump(u, n) * dS, dg.jump(dg.jump(u), n) * dS),
    (4*jump(u, n) * dS, dg.jump(dg.jump(dg.jump(u)), n) * dS),
    # ---
    # jump(jump(u)*avg(u), n) =  2*jump(u, n)*avg(u)
    # (2*(jump(u, n) + jump(u))**2 * dS, dg.jump(dg.jump(u, n) + jump(u))**2 * dS),  # TODO
)


@pytest.mark.parametrize("comparison", vector_facet_normal_comparisons)
def test_vector_with_facet_normal(comparison):
    assert assemble(comparison[0]) == pytest.approx(assemble(dg.apply_dg_operator_lowering(comparison[1])), rel=1e-15)


const_mat = Constant(((2.0, 4.0), (-4.0, 1.0)))
ident = Identity(2)


def tensor_jump(u, n):
    return outer(u, n)("+") + outer(u, n)("-")


tensor_facet_normal_comparisons = (
    # Standard jump rules
    (tensor_jump(u, n)**2*dS, dg.tensor_jump(u, n)**2*dS),
    (tensor_jump(u, n)**2*dS, dg.avg(dg.tensor_jump(u, n)**2)*dS),
    (tensor_jump(u, n)**2*dS, dg.avg(dg.avg(dg.tensor_jump(u, n)**2))*dS),
    # Commutitive
    (tensor_jump(2*u, n)**2*dS, dg.tensor_jump(2*u, n)**2*dS),
    (inner(tensor_jump(2*u, n), tensor_jump(4*u, n))*dS, inner(dg.tensor_jump(2*u, n), dg.tensor_jump(4*u, n))*dS),
    (inner(ident, tensor_jump(4*u, n))*dS, inner(ident, dg.avg(dg.tensor_jump(4*u, n)))*dS),
    (inner(const_mat, tensor_jump(4*u, n))*dS, inner(const_mat, dg.tensor_jump(4*u, n))*dS),
    (inner(const_mat, tensor_jump(4*u, n))*dS, inner(const_mat, dg.avg(dg.tensor_jump(4*u, n)))*dS),
    # Distributive
    (tensor_jump(2*u + u, n)**2*dS, dg.tensor_jump(2*u + u, n)**2*dS),
    (tensor_jump(2*u, n)**2*dS, dg.tensor_jump(2*u + dg.avg(u), n)**2*dS),
    (tensor_jump(2*u, n)**2*dS, dg.tensor_jump(2*u + dg.avg(ident*u), n)**2*dS),
    (tensor_jump(2*u, n)**2*dS, dg.tensor_jump(2*u + dg.avg(ident*u), n)**2*dS),
)


@pytest.mark.parametrize("comparison", tensor_facet_normal_comparisons)
def test_tensor_with_facet_normal(comparison):
    assert assemble(comparison[0]) == pytest.approx(assemble(dg.apply_dg_operator_lowering(comparison[1])), rel=1e-15)


def F_v(u, grad_u):
    return (Constant(1.0)*u[1] + u[0]**2/u[1])*grad_u


G = dolfin_dg.homogeneity_tensor(F_v, u)

mock_form_comparisons = (
    (inner(tensor_jump(u, n), avg(dolfin_dg.hyper_tensor_T_product(G, grad(v))))*dS,
     inner(dg.tensor_jump(u, n), dg.avg(dolfin_dg.hyper_tensor_T_product(G, grad(v))))*dS),
    (inner(avg(F_v(u, grad(u))), tensor_jump(v, n))*dS,
     inner(dg.avg(F_v(u, grad(u))), dg.tensor_jump(v, n))*dS),
    (inner(dolfin_dg.hyper_tensor_product(avg(G), tensor_jump(u, n)), tensor_jump(v, n))*dS,
     inner(dolfin_dg.hyper_tensor_product(dg.avg(G), dg.tensor_jump(u, n)), dg.tensor_jump(v, n))*dS)
)

@pytest.mark.parametrize("comparison", mock_form_comparisons)
def test_mock_forms_vector_pde(comparison):
    vufl = assemble(comparison[0])
    vdg = assemble(dg.apply_dg_operator_lowering(comparison[1]))
    assert vufl.norm("l2") == pytest.approx(vdg.norm("l2"), rel=1e-15)


@pytest.mark.parametrize("comparison", mock_form_comparisons)
def test_mock_forms_matrix_assembly_pde(comparison):
    aufl = derivative(comparison[0], u)
    adg = derivative(dg.apply_dg_operator_lowering(comparison[1]), u)

    Aufl = assemble(aufl)
    Adg = assemble(adg)

    assert Aufl.norm("frobenius") == pytest.approx(Adg.norm("frobenius"), rel=1e-15)


def test_zeros():
    zero_exprs = (
        # Avg
        dg.avg(dg.jump(u)),
        # Jump
        dg.jump(dg.avg(u)),
        dg.jump(dg.jump(u, n)),
        # NormalJump
        dg.jump(dg.jump(u, n), n),
        dg.jump(dg.avg(u), n),
        # -- product
        dg.jump(dg.jump(u, n)*dg.jump(u, n), n)
    )

    for zero_expr in zero_exprs:
        zer = dg.apply_dg_operator_lowering(zero_expr)
        assert isinstance(zer, ufl.constantvalue.Zero)

