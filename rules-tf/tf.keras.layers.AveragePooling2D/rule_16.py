import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When using "valid" padding, the spatial dimensions of the input must be greater than or equal to pool_size (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == 21, v["arg3_value"] == 24), And(Select(v["arg4_shape"], 1) >= v["arg1_value"], Select(v["arg4_shape"], 2) >= v["arg1_value"]), If(And(v["arg2_value"] == 21, v["arg3_value"] == 25), And(Select(v["arg4_shape"], 2) >= v["arg1_value"], Select(v["arg4_shape"], 3) >= v["arg1_value"]), True))) if n else
          If(And(v["arg2_value"] == 21, v["arg3_value"] == 24), And(Select(v["arg4_shape"], 1) >= v["arg1_value"], Select(v["arg4_shape"], 2) >= v["arg1_value"]), If(And(v["arg2_value"] == 21, v["arg3_value"] == 25), And(Select(v["arg4_shape"], 2) >= v["arg1_value"], Select(v["arg4_shape"], 3) >= v["arg1_value"]), True)))
)

def rule_16_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = String('arg3_value')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 16
        rule_16(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape']}, neg)
