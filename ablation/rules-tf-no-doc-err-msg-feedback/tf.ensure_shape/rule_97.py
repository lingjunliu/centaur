import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Tensor v_1's shape at v_2 must be same as the length of list v_3 if string v_4 equals "channel_first" or "channel_last" (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg4_value"] == 25, v["arg4_value"] == 24), Select(v["arg1_shape"], v["arg2_value"]) == v["arg3_length"], True)) if n else
          If(Or(v["arg4_value"] == 25, v["arg4_value"] == 24), Select(v["arg1_shape"], v["arg2_value"]) == v["arg3_length"], True))
)

def rule_97_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, list) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg4_value = String('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 97
        rule_97(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg4_value': arg4['value']}, neg)
