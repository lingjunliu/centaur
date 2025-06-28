import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If float v_1 is positive, and string v_2 is equal to "max", then the absolute value of minimum of tensor v_3 must be less than 10 (Rule 399)

rule_399 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, v["arg2_value"] == 9), (If(Select(v["arg3_range"], 0) < 0, 0 - Select(v["arg3_range"], 0), Select(v["arg3_range"], 0))) < 10, False)) if n else
          If(And(v["arg1_value"] > 0, v["arg2_value"] == 9), (If(Select(v["arg3_range"], 0) < 0, 0 - Select(v["arg3_range"], 0), Select(v["arg3_range"], 0))) < 10, False))
)

def rule_399_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = String('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values.index(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 399
        rule_399(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_399(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range']}, neg)
