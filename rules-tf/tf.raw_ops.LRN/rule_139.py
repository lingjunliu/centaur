import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if depth_radius is high then bias must be less than 10 and greater than zero, and alpha must be within reasonable bounds (less than 100 (Rule 139)

rule_139 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 7, And(And(And(And(And(v["arg2_value"] < 10, v["arg2_value"] > 0), v["arg3_value"] < 100), v["arg3_value"] > 0), (Or(Select(v["arg4_shape"], 2) > 5, Select(v["arg4_shape"], 3) > 5))), (Or(Or(Select(v["arg4_shape"], 0) > 1, Select(v["arg4_shape"], 1) > 1), Select(v["arg4_shape"], 2) > 1))), True)) if n else
          If(v["arg1_value"] > 7, And(And(And(And(And(v["arg2_value"] < 10, v["arg2_value"] > 0), v["arg3_value"] < 100), v["arg3_value"] > 0), (Or(Select(v["arg4_shape"], 2) > 5, Select(v["arg4_shape"], 3) > 5))), (Or(Or(Select(v["arg4_shape"], 0) > 1, Select(v["arg4_shape"], 1) > 1), Select(v["arg4_shape"], 2) > 1))), True))
)

def rule_139_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 139
        rule_139(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_139(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape']}, neg)
