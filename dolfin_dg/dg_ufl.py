import functools

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
    return NormalJump(u, n)


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
class Avg(Restricted):
    __slots__ = ()

    def __init__(self, f):
        Operator.__init__(self, (f,))

    def __str__(self):
        return "〈%s〉" % parstr(self.ufl_operands[0], self)


@ufl_type(is_abstract=True,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0,
          is_restriction=False)
class Jump(Restricted, CompoundTensorOperator):
    __slots__ = ()

    def __init__(self, f):
        Operator.__init__(self, (f,))

    def side(self):
        return self._side

    def evaluate(self, x, mapping, component, index_values):
        return self.ufl_operands[0].evaluate(x, mapping, component,
                                             index_values)

    def __str__(self):
        return "[" + str(self.ufl_operands[0]) + "]"


@ufl_type(is_abstract=True,
          inherit_indices_from_operand=0,
          is_restriction=False)
class NormalJump(Restricted, CompoundTensorOperator):
    __slots__ = ()

    def __init__(self, f, n):
        Operator.__init__(self, (f, n))

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
        v, n = self.ufl_operands
        if len(v.ufl_shape) == 0:
            return n.ufl_shape
        return ufl.dot(v, n).ufl_shape


@ufl_type(num_ops=2)
class TensorJump(NormalJump):
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
class TangentJump(NormalJump):
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
        o = apply_average_lowering(o)
        return o

    def jump(self, o):
        o_args = o.ufl_operands
        v = apply_jump_lowering(o_args[0])

        if isinstance(v, Zero):
            return v

        return v("+") - v("-")

    def normal_jump(self, o):
        n = o.ufl_operands[1]
        v = apply_jump_lowering(o.ufl_operands[0])

        r = len(v.ufl_shape)

        if isinstance(v, Zero):
            if r == 0:
                return v * n
            return ufl.dot(v, n)

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
        o_arg = o.ufl_operands[0]

        # Handle {u + u} = {u} + {u}
        if isinstance(o_arg, ufl.algebra.Sum):
            distributed_avg = map(lambda oarg: apply_average_lowering(Avg(oarg)), o_arg.ufl_operands)
            return ufl.algebra.Sum(*distributed_avg)

        # Handle {u {u}} = {u} {u}
        if isinstance(o_arg, ufl.algebra.Product):
            avg_components = [o_arg_arg for o_arg_arg in o_arg.ufl_operands if isinstance(o_arg_arg, (Avg, Jump))]
            if len(avg_components) > 0:
                avg_components = map(apply_average_lowering, avg_components)
                avg_components = functools.reduce(ufl.algebra.Product, avg_components)
                other_components = [o_arg_arg for o_arg_arg in o_arg.ufl_operands if not isinstance(o_arg_arg, (Avg, Jump))]
                if len(other_components) > 0:
                    other_components = functools.reduce(ufl.algebra.Product, other_components)
                    other_components = apply_average_lowering(Avg(other_components))
                    return ufl.algebra.Product(other_components, avg_components)
                return avg_components

        reto = apply_average_lowering(o.ufl_operands[0])
        if reto == o.ufl_operands[0]:
            return 0.5 * (reto("+") + reto("-"))
        return reto

    def jump(self, o):
        return Zero(shape=o.ufl_shape)

    def normal_jump(self, o):
        lowered_jump = apply_normal_jump_lowering(o.ufl_operands[0])
        if isinstance(lowered_jump, Zero):
            return lowered_jump

        n = o.ufl_operands[1]
        v = lowered_jump
        r = len(v.ufl_shape)
        if r == 0:
            jump_eval = v('+') * n('+') + v('-') * n('-')
        else:
            jump_eval = ufl.dot(v('+'), n('+')) + ufl.dot(v('-'), n('-'))
        return jump_eval

    def tensor_jump(self, o):
        lowered_jump = apply_normal_jump_lowering(o.ufl_operands[0])
        if isinstance(lowered_jump, Zero):
            return lowered_jump

        n = o.ufl_operands[1]
        v = lowered_jump
        return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")

    def tangent_jump(self, o):
        # jump_class = type(o)
        lowered_jump = apply_normal_jump_lowering(o.ufl_operands[0])
        if isinstance(lowered_jump, Zero):
            return lowered_jump

        n = o.ufl_operands[1]
        v = lowered_jump
        return dg_cross(n("+"), v("+")) + dg_cross(n("-"), v("-"))


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
        lowered_jump = apply_jump_lowering(o.ufl_operands[0])
        return 2*lowered_jump

    def normal_jump(self, o):
        return Zero(shape=o.ufl_shape)
        # v, n = o.ufl_operands
        # lowered_jump = apply_normal_jump_lowering(v)
        # if len(v.ufl_shape) == 0:
        #     return 2*lowered_jump*n
        # return 2*ufl.dot(lowered_jump, n)
    #
    # def tensor_jump(self, o):
    #     return Zero(shape=o.ufl_shape)
    #
    # def tangent_jump(self, o):
    #     return Zero(shape=o.ufl_shape)


def apply_jump_lowering(expression):
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = JumpLowering()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class NormalJumpLowering(MultiFunction):

    operator = MultiFunction.reuse_if_untouched

    def _ignore(self, o):
        return o

    terminal = _ignore

    def avg(self, o):
        return Zero(shape=o.ufl_shape)

    def jump(self, o):
        lowered_jump = apply_jump_lowering(o.ufl_operands[0])
        return 2*lowered_jump

    def normal_jump(self, o):
        return Zero(shape=ufl.dot(o.ufl_operands[0], o.ufl_operands[1]))

    def tensor_jump(self, o):
        o_args = o.ufl_operands
        return Zero(shape=ufl.outer(o_args[0], o_args[1]).ufl_shape)

    def tangent_jump(self, o):
        o_args = o.ufl_operands
        return Zero(shape=ufl.cross(o_args[1], o_args[0]).ufl_shape)


def apply_normal_jump_lowering(expression):
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = NormalJumpLowering()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)

# class DGRestrictionDispatcher(MultiFunction):
#
#     operator = MultiFunction.reuse_if_untouched
#     terminal = MultiFunction.reuse_if_untouched
#
#     def restricted(self, o):
#         flip = {"+": "-", "-": "+"}
#
#         parent_side = o.side()
#
#         print("Top parent:", o, "parent side", parent_side)
#         ret_val = map_expr_dag(_rp[parent_side], o.ufl_operands[0])
#
#         if not isinstance(ret_val, Restricted):
#             print("Top parent, ret_val is not restricted", ret_val)
#             return ret_val#(parent_side)
#
#         child_side = ret_val.side()
#
#         # new_side = child_side
#         # if parent_side == "-":
#         #     new_side = flip[ret_val.side()]
#
#         print("Top parent: Returning", ret_val, "parent side", parent_side, "child_side", child_side)#, "new_side", new_side)
#         return ret_val
#
#
# class DGRestrictionApplier(MultiFunction):
#     def __init__(self, side):
#         MultiFunction.__init__(self)
#         self.parent_side = side
#
#     def terminal(self, o):
#         return o
#
#     # Default: Operators should reconstruct only if subtrees are not touched
#     operator = MultiFunction.reuse_if_untouched
#
#     def coefficient(self, o):
#         print("Applier: coefficient", o, "parent side", self.parent_side)
#         return o(self.parent_side)
#
#     def sum(self, o):
#         print("Applier: sum", o, "opands", *o.ufl_operands, "parent side", self.parent_side)
#         side = self.parent_side
#         ret_val = map(lambda opand: map_expr_dag(_rp[side], opand), o.ufl_operands)
#         return ufl.algebra.Sum(*ret_val)
#
#     # Apply restriction coming back up
#     def restricted(self, o):
#         flip = {"+": "-", "-": "+"}
#
#         print("Applier: restricted", o, "old child side", o.side(), "parent_side", self.parent_side)
#         ret_val = map_expr_dag(_rp[o.side()], o.ufl_operands[0])
#
#         if not isinstance(ret_val, Restricted):
#             new_side = o.side()
#             if self.parent_side == "-":
#                 new_side = flip[o.side()]
#             print("Applier: non restricted flip? parent side", self.parent_side, "child side", o.side(), "new side", new_side)
#             print("Applier: returning non restricted", ret_val(new_side))
#             return ret_val#(new_side)
#
#         new_child_side = ret_val.side()
#         new_side = new_child_side
#         if self.parent_side == "-":
#             new_side = flip[new_child_side]
#
#         print("Applier: restricted flip? parent side", self.parent_side, "new child side",new_child_side, "new side", new_side)
#         print("Applier: returning restricted", ret_val.ufl_operands[0](new_side))
#         return ret_val.ufl_operands[0]#(new_side)
#
#
#
# _rp = {"+": DGRestrictionApplier("+"),
#        "-": DGRestrictionApplier("-")}
#
#
# def apply_dg_restrictions(expression):
#     """Some terminals can be restricted from either side.
#
#     This applies a default restriction to such terminals if unrestricted."""
#     integral_types = [k for k in integral_type_to_measure_name.keys()
#                       if k.startswith("interior_facet")]
#     rules = DGRestrictionDispatcher()
#     return map_integrand_dags(rules, expression,
#                               only_integral_type=integral_types)
