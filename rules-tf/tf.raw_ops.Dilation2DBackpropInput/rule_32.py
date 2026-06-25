import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When padding is not VALID or SAME, then specific calculations/conditions must still hold. (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg6_value"] != 28, v["arg6_value"] != 29), And(Select(v["arg3_shape"], 1) == (Select(v["arg1_shape"], 1) - (Select(v["arg2_shape"], 0) + (Select(v["arg2_shape"], 0) - 1) * (Select(v["arg5_values"], 1) - 1)) + Select(v["arg4_values"], 1)) / Select(v["arg4_values"], 1), Select(v["arg3_shape"], 2) == (Select(v["arg1_shape"], 2) - (Select(v["arg2_shape"], 1) + (Select(v["arg2_shape"], 1) - 1) * (Select(v["arg5_values"], 2) - 1)) + Select(v["arg4_values"], 2)) / Select(v["arg4_values"], 2)), True)) if n else
          If(And(v["arg6_value"] != 28, v["arg6_value"] != 29), And(Select(v["arg3_shape"], 1) == (Select(v["arg1_shape"], 1) - (Select(v["arg2_shape"], 0) + (Select(v["arg2_shape"], 0) - 1) * (Select(v["arg5_values"], 1) - 1)) + Select(v["arg4_values"], 1)) / Select(v["arg4_values"], 1), Select(v["arg3_shape"], 2) == (Select(v["arg1_shape"], 2) - (Select(v["arg2_shape"], 1) + (Select(v["arg2_shape"], 1) - 1) * (Select(v["arg5_values"], 2) - 1)) + Select(v["arg4_values"], 2)) / Select(v["arg4_values"], 2)), True))
)

def rule_32_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

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

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_values = Array('arg5_values', IntSort(), IntSort())
        arg6_value = Int('arg6_value')

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

        # Constraints for rule 32
        rule_32(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_values': arg4_values, 'arg5_values': arg5_values, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_values': arg4['values'], 'arg5_values': arg5['values'], 'arg6_value': arg6['value']}, neg)
