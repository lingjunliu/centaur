import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The scale cannot be too small relative to the range of input values to prevent information loss (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] > (Select(v["arg2_range"], 1) - Select(v["arg2_range"], 0)) / (v["arg4_value"] - v["arg3_value"]) / 1000) if n else
          v["arg1_value"] > (Select(v["arg2_range"], 1) - Select(v["arg2_range"], 0)) / (v["arg4_value"] - v["arg3_value"]) / 1000)
)

def rule_65_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
