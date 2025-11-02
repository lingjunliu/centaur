import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Absolute difference of two floats v_1 and v_2 must be less than the min of tensor v_3 and max of v_3 (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] - v["arg2_value"] > 0, And(v["arg1_value"] - v["arg2_value"] < Select(v["arg3_range"], 0), v["arg1_value"] - v["arg2_value"] < Select(v["arg3_range"], 1)), And(v["arg2_value"] - v["arg1_value"] < Select(v["arg3_range"], 0), v["arg2_value"] - v["arg1_value"] < Select(v["arg3_range"], 1)))) if n else
          If(v["arg1_value"] - v["arg2_value"] > 0, And(v["arg1_value"] - v["arg2_value"] < Select(v["arg3_range"], 0), v["arg1_value"] - v["arg2_value"] < Select(v["arg3_range"], 1)), And(v["arg2_value"] - v["arg1_value"] < Select(v["arg3_range"], 0), v["arg2_value"] - v["arg1_value"] < Select(v["arg3_range"], 1))))
)

def rule_76_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 76
        rule_76(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range']}, neg)
