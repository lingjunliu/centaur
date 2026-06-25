import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If padding is EXPLICIT, then explicit_paddings must have length 8, be non-negative, and less than input height and width from input_sizes, out_backprop should be a 4d tensor (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 30, And(v["arg2_length"] == 8, And([Implies(i < (v["arg2_length"] - 1 + 1), And(And(And(Select(v["arg2_values"], i) >= 0, Select(v["arg2_values"], 0) + Select(v["arg2_values"], 1) < Select(v["arg4_shape"], 1)), Select(v["arg2_values"], 2) + Select(v["arg2_values"], 3) < Select(v["arg4_shape"], 2)), v["arg3_ndim"] == 4)) for i in range(6)])), True)) if n else
          If(v["arg1_value"] == 30, And(v["arg2_length"] == 8, And([Implies(i < (v["arg2_length"] - 1 + 1), And(And(And(Select(v["arg2_values"], i) >= 0, Select(v["arg2_values"], 0) + Select(v["arg2_values"], 1) < Select(v["arg4_shape"], 1)), Select(v["arg2_values"], 2) + Select(v["arg2_values"], 3) < Select(v["arg4_shape"], 2)), v["arg3_ndim"] == 4)) for i in range(6)])), True))
)

def rule_58_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape']}, neg)
