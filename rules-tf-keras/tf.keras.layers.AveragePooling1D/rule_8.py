import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# valid output shape constraint for valid padding (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == 21, v["arg4_value"] == 24), Select(v["arg5_shape"], 1) - v["arg1_value"] + 1 >= v["arg2_value"], If(And(v["arg3_value"] == 21, v["arg4_value"] == 25), Select(v["arg5_shape"], 2) - v["arg1_value"] + 1 >= v["arg2_value"], True))) if n else
          If(And(v["arg3_value"] == 21, v["arg4_value"] == 24), Select(v["arg5_shape"], 1) - v["arg1_value"] + 1 >= v["arg2_value"], If(And(v["arg3_value"] == 21, v["arg4_value"] == 25), Select(v["arg5_shape"], 2) - v["arg1_value"] + 1 >= v["arg2_value"], True)))
)

def rule_8_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')
        arg4_value = String('arg4_value')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 8
        rule_8(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_shape': arg5['shape']}, neg)
