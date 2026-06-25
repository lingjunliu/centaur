import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data_format is NCHW and padding is VALID, then the height and width dimensions of out_backprop are computed as (input_height - filter_height + 1 + (dilation_height - 1 (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg7_value"] == 34, v["arg6_value"] == 28), And(Select(v["arg2_shape"], 2) == (Select(v["arg1_shape"], 2) - Select(v["arg3_shape"], 0) + 1 + (Select(v["arg5_values"], 2) - 1) * (Select(v["arg3_shape"], 0) - 1)) / Select(v["arg4_values"], 2), Select(v["arg2_shape"], 3) == (Select(v["arg1_shape"], 3) - Select(v["arg3_shape"], 1) + 1 + (Select(v["arg5_values"], 3) - 1) * (Select(v["arg3_shape"], 1) - 1)) / Select(v["arg4_values"], 3)), True)) if n else
          If(And(v["arg7_value"] == 34, v["arg6_value"] == 28), And(Select(v["arg2_shape"], 2) == (Select(v["arg1_shape"], 2) - Select(v["arg3_shape"], 0) + 1 + (Select(v["arg5_values"], 2) - 1) * (Select(v["arg3_shape"], 0) - 1)) / Select(v["arg4_values"], 2), Select(v["arg2_shape"], 3) == (Select(v["arg1_shape"], 3) - Select(v["arg3_shape"], 1) + 1 + (Select(v["arg5_values"], 3) - 1) * (Select(v["arg3_shape"], 1) - 1)) / Select(v["arg4_values"], 3)), True))
)

def rule_74_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False
        if not isinstance(arg6, str):
            return False
        if not isinstance(arg7, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_values = Array('arg5_values', IntSort(), IntSort())
        arg6_value = Int('arg6_value')
        arg7_value = Int('arg7_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])
        solver.add(arg6_value == list_of_string_values_tf.index(arg6))
        solver.add(arg7_value == list_of_string_values_tf.index(arg7))

        # Constraints for rule 74
        rule_74(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_values': arg4_values, 'arg5_values': arg5_values, 'arg6_value': arg6_value, 'arg7_value': arg7_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_values': arg4['values'], 'arg5_values': arg5['values'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value']}, neg)
