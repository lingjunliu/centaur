import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# conv1d input width must support dilated filter width (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 26, (If(v["arg5_value"] == 29, (If(v["arg4_length"] == 1, Select(v["arg1_shape"], 1) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 0), If(v["arg4_length"] == 3, Select(v["arg1_shape"], 1) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 1), True))), If(v["arg5_value"] == 30, (If(v["arg4_length"] == 1, Select(v["arg1_shape"], 2) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 0), If(v["arg4_length"] == 3, Select(v["arg1_shape"], 2) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 1), True))), True))), True)) if n else
          If(v["arg3_value"] == 26, (If(v["arg5_value"] == 29, (If(v["arg4_length"] == 1, Select(v["arg1_shape"], 1) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 0), If(v["arg4_length"] == 3, Select(v["arg1_shape"], 1) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 1), True))), If(v["arg5_value"] == 30, (If(v["arg4_length"] == 1, Select(v["arg1_shape"], 2) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 0), If(v["arg4_length"] == 3, Select(v["arg1_shape"], 2) >= 1 + (Select(v["arg2_shape"], 0) - 1) * Select(v["arg4_values"], 1), True))), True))), True))
)

def rule_138_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_value = Int('arg5_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 138
        rule_138(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_values': arg4_values, 'arg4_length': arg4_length, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length'], 'arg5_value': arg5['value']}, neg)
