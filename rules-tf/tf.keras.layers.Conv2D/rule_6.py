import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input channels and filters must both be divisible by groups (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(And(v["arg3_value"] > 0, (If(v["arg2_value"] == 24, And(Select(v["arg1_shape"], 3) % v["arg3_value"] == 0, v["arg4_value"] % v["arg3_value"] == 0), (If(v["arg2_value"] == 25, And(Select(v["arg1_shape"], 1) % v["arg3_value"] == 0, v["arg4_value"] % v["arg3_value"] == 0), True)))))) if n else
          And(v["arg3_value"] > 0, (If(v["arg2_value"] == 24, And(Select(v["arg1_shape"], 3) % v["arg3_value"] == 0, v["arg4_value"] % v["arg3_value"] == 0), (If(v["arg2_value"] == 25, And(Select(v["arg1_shape"], 1) % v["arg3_value"] == 0, v["arg4_value"] % v["arg3_value"] == 0), True))))))
)

def rule_6_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 6
        rule_6(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
