import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When Saturation Factor equals 1 and each pixel of the image have values in between 0 and 1, then output should also have the same range (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg2_value"] == 1.0, Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) <= 1), And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= 1), True)) if n else
          If(And(And(v["arg2_value"] == 1.0, Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) <= 1), And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= 1), True))
)

def rule_62_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 62
        rule_62(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
