import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data format is channels first and dilations are 1,1,1,1 then the output height is (input_height + 2 * pad_height - filter_height - (filter_height - 1 (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(v["arg5_value"] == 25, Select(v["arg2_values"], 0) == 1), Select(v["arg2_values"], 1) == 1), Select(v["arg2_values"], 2) == 1), Select(v["arg2_values"], 3) == 1), (Select(v["arg1_shape"], 2) + 2 * Select(v["arg3_values"], 0) - Select(v["arg4_shape"], 0) - (Select(v["arg4_shape"], 0) - 1) * (Select(v["arg2_values"], 2) - 1)) / Select(v["arg3_values"], 2) + 1 > 0, True)) if n else
          If(And(And(And(And(v["arg5_value"] == 25, Select(v["arg2_values"], 0) == 1), Select(v["arg2_values"], 1) == 1), Select(v["arg2_values"], 2) == 1), Select(v["arg2_values"], 3) == 1), (Select(v["arg1_shape"], 2) + 2 * Select(v["arg3_values"], 0) - Select(v["arg4_shape"], 0) - (Select(v["arg4_shape"], 0) - 1) * (Select(v["arg2_values"], 2) - 1)) / Select(v["arg3_values"], 2) + 1 > 0, True))
)

def rule_45_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_value = String('arg5_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 45
        rule_45(solver, {'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg3_values': arg3_values, 'arg4_shape': arg4_shape, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values'], 'arg4_shape': arg4['shape'], 'arg5_value': arg5['value']}, neg)
