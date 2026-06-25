import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# For EXPLICIT padding, output tensor spatial dimensions must be greater than 0 (Rule 90)

rule_90 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 30, If(v["arg1_value"] == 33, And((Select(v["arg4_shape"], 1) + Select(v["arg3_values"], 0) + Select(v["arg3_values"], 1) - (Select(v["arg7_values"], 1) - 1) * (Select(v["arg5_shape"], 0) - 1) - Select(v["arg5_shape"], 0) + Select(v["arg6_values"], 1)) / Select(v["arg6_values"], 1) > 0, (Select(v["arg4_shape"], 2) + Select(v["arg3_values"], 2) + Select(v["arg3_values"], 3) - (Select(v["arg7_values"], 2) - 1) * (Select(v["arg5_shape"], 1) - 1) - Select(v["arg5_shape"], 1) + Select(v["arg6_values"], 2)) / Select(v["arg6_values"], 2) > 0), If(v["arg1_value"] == 34, And((Select(v["arg4_shape"], 2) + Select(v["arg3_values"], 0) + Select(v["arg3_values"], 1) - (Select(v["arg7_values"], 2) - 1) * (Select(v["arg5_shape"], 0) - 1) - Select(v["arg5_shape"], 0) + Select(v["arg6_values"], 2)) / Select(v["arg6_values"], 2) > 0, (Select(v["arg4_shape"], 3) + Select(v["arg3_values"], 2) + Select(v["arg3_values"], 3) - (Select(v["arg7_values"], 3) - 1) * (Select(v["arg5_shape"], 1) - 1) - Select(v["arg5_shape"], 1) + Select(v["arg6_values"], 3)) / Select(v["arg6_values"], 3) > 0), True)), True)) if n else
          If(v["arg2_value"] == 30, If(v["arg1_value"] == 33, And((Select(v["arg4_shape"], 1) + Select(v["arg3_values"], 0) + Select(v["arg3_values"], 1) - (Select(v["arg7_values"], 1) - 1) * (Select(v["arg5_shape"], 0) - 1) - Select(v["arg5_shape"], 0) + Select(v["arg6_values"], 1)) / Select(v["arg6_values"], 1) > 0, (Select(v["arg4_shape"], 2) + Select(v["arg3_values"], 2) + Select(v["arg3_values"], 3) - (Select(v["arg7_values"], 2) - 1) * (Select(v["arg5_shape"], 1) - 1) - Select(v["arg5_shape"], 1) + Select(v["arg6_values"], 2)) / Select(v["arg6_values"], 2) > 0), If(v["arg1_value"] == 34, And((Select(v["arg4_shape"], 2) + Select(v["arg3_values"], 0) + Select(v["arg3_values"], 1) - (Select(v["arg7_values"], 2) - 1) * (Select(v["arg5_shape"], 0) - 1) - Select(v["arg5_shape"], 0) + Select(v["arg6_values"], 2)) / Select(v["arg6_values"], 2) > 0, (Select(v["arg4_shape"], 3) + Select(v["arg3_values"], 2) + Select(v["arg3_values"], 3) - (Select(v["arg7_values"], 3) - 1) * (Select(v["arg5_shape"], 1) - 1) - Select(v["arg5_shape"], 1) + Select(v["arg6_values"], 3)) / Select(v["arg6_values"], 3) > 0), True)), True))
)

def rule_90_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not (isinstance(arg6, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg6)):
            return False
        if not (isinstance(arg7, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg7)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())
        arg6_values = Array('arg6_values', IntSort(), IntSort())
        arg7_values = Array('arg7_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])
        for i in range(len(arg6)):
            arg6_values = Store(arg6_values, i, arg6[i])
        for i in range(len(arg7)):
            arg7_values = Store(arg7_values, i, arg7[i])

        # Constraints for rule 90
        rule_90(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_values': arg3_values, 'arg4_shape': arg4_shape, 'arg5_shape': arg5_shape, 'arg6_values': arg6_values, 'arg7_values': arg7_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_90(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values'], 'arg4_shape': arg4['shape'], 'arg5_shape': arg5['shape'], 'arg6_values': arg6['values'], 'arg7_values': arg7['values']}, neg)
