import ufl
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction
from ufl.constantvalue import Zero
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.index_combination_utils import merge_nonoverlapping_indices
from ufl.measure import integral_type_to_measure_name
from ufl.precedence import parstr
from ufl.tensoralgebra import CompoundTensorOperator


def dg_cross(u, v):
    if len(u.ufl_shape) == 0 or len(v.ufl_shape) == 0:
        raise TypeError("Input argument must be a vector")
    assert(len(u.ufl_shape) == 1 and len(v.ufl_shape) == 1)
    if u.ufl_shape[0] == 2 and v.ufl_shape[0] == 2:
        return u[0]*v[1] - u[1]*v[0]
    return ufl.cross(u, v)


def avg(u):
    u = ufl.as_ufl(u)
    return Avg(u)


def jump(u, n=None):
    u = ufl.as_ufl(u)
    if n is None:
        return Jump(u)
    n = ufl.as_ufl(n)
    return Jump(u, n)


def tensor_jump(u, n):
    u = ufl.as_ufl(u)
    n = ufl.as_ufl(n)
    return TensorJump(u, n)


def tangent_jump(u, n):
    if len(u.ufl_shape) == 0:
        raise TypeError("Input argument must be a vector")
    assert(len(u.ufl_shape) == 1)
    assert(u.ufl_shape[0] in (2, 3))
    return TangentJump(u, n)


@ufl_type(is_abstract=True,
          num_ops=1,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0,
          is_restriction=False,
          is_terminal_modifier=True)
class Avg(Operator):
    __slots__ = ()

    def __init__(self, f):
        Operator.__init__(self, (f,))

    def __str__(self):
        return "{%s}" % parstr(self.ufl_operands[0], self)


@ufl_type(is_abstract=True,
          inherit_indices_from_operand=0,
          is_restriction=False)
class Jump(CompoundTensorOperator):
    __slots__ = ()

    def __init__(self, f, *n):
        Operator.__init__(self, (f, *n))

    def side(self):
        return self._side

    def evaluate(self, x, mapping, component, index_values):
        return self.ufl_operands[0].evaluate(x, mapping, component,
                                             index_values)

    def __str__(self):
        return "〚" + " ⋅ ".join(map(lambda o: parstr(o, self),
                                    self.ufl_operands)) + "〛"

    @property
    def ufl_shape(self):
        if len(self.ufl_operands) == 1:
            return self.ufl_operands[0].ufl_shape
        if len(self.ufl_operands[0].ufl_shape) == 0:
            return self.ufl_operands[1].ufl_shape
        return ufl.dot(self.ufl_operands[0], self.ufl_operands[1]).ufl_shape


@ufl_type(num_ops=2)
class TensorJump(Jump):
    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        ash, bsh = a.ufl_shape, b.ufl_shape
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(ash + bsh, fi, fid)
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, f, n):
        CompoundTensorOperator.__init__(self, (f, n))
        fi, fid = merge_nonoverlapping_indices(f, n)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    def __str__(self):
        o1, o2 = parstr(self.ufl_operands[0], self), \
                 parstr(self.ufl_operands[1], self)
        return "〚%s ⊗ %s〛" % (o1, o2)

    @property
    def ufl_shape(self):
        return ufl.outer(self.ufl_operands[0], self.ufl_operands[1]).ufl_shape


@ufl_type(num_ops=2)
class TangentJump(Jump):
    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        ash, bsh = a.ufl_shape, b.ufl_shape
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(ash + bsh, fi, fid)
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, f, n):
        CompoundTensorOperator.__init__(self, (f, n))
        fi, fid = merge_nonoverlapping_indices(f, n)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    def __str__(self):
        return "〚%s × %s〛" % \
               (parstr(self.ufl_operands[0], self),
                parstr(self.ufl_operands[1], self))

    @property
    def ufl_shape(self):
        return dg_cross(self.ufl_operands[0], self.ufl_operands[1]).ufl_shape


class DGOperatorLowering(MultiFunction):

    def _ignore(self, o):
        return o

    terminal = _ignore
    operator = MultiFunction.reuse_if_untouched

    def avg(self, o):
        o = apply_average_lowering(o.ufl_operands[0])
        if isinstance(o, Jump):
            return apply_dg_operators(o)
        return 0.5 * (o("+") + o("-"))

    def jump(self, o):
        o_args = o.ufl_operands
        v = apply_jump_lowering(o_args[0])
        if len(o_args) == 1:
            return v("+") - v("-")
        assert len(o_args) == 2
        n = o_args[1]
        r = len(v.ufl_shape)
        if r == 0:
            jump_eval = v('+') * n('+') + v('-') * n('-')
        else:
            jump_eval = ufl.dot(v('+'), n('+')) + ufl.dot(v('-'), n('-'))
        return jump_eval

    def tensor_jump(self, o):
        n = o.ufl_operands[1]
        v = apply_jump_lowering(o.ufl_operands[0])
        return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")

    def tangent_jump(self, o):
        n = o.ufl_operands[1]
        v = apply_jump_lowering(o.ufl_operands[0])
        return dg_cross(n("+"), v("+")) + dg_cross(n("-"), v("-"))


def apply_dg_operators(expression):
    "Propagate restriction nodes to wrap differential terminals directly."
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = DGOperatorLowering()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class AverageLowering(MultiFunction):

    operator = MultiFunction.reuse_if_untouched

    def _ignore(self, o):
        return o

    terminal = _ignore

    def avg(self, o):
        return apply_average_lowering(o.ufl_operands[0])

    def jump(self, o):
        if len(o.ufl_operands) == 0:
            return Jump(apply_jump_lowering(o.ufl_operands[0]))
        return Jump(apply_jump_lowering(o.ufl_operands[0]), o.ufl_operands[1])


def apply_average_lowering(expression):
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = AverageLowering()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class JumpLowering(MultiFunction):

    operator = MultiFunction.reuse_if_untouched

    def _ignore(self, o):
        return o

    terminal = _ignore

    def avg(self, o):
        return Zero(shape=o.ufl_shape)

    def jump(self, o):
        return Zero(shape=o.ufl_shape)


def apply_jump_lowering(expression):
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = JumpLowering()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)
