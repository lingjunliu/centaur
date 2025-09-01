import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the image has a large dynamic range, then gamma should be close to 1 (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) > 100, And(v["arg2_value"] > 0.9, v["arg2_value"] < 1.1), True)) if n else
          If(Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) > 100, And(v["arg2_value"] > 0.9, v["arg2_value"] < 1.1), True))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 52
        rule_52(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
