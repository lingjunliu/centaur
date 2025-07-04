import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# threshold must be greater than or equal to the minimum value of the input tensor if value is set to min value (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == Select(v["arg1_range"], 0), v["arg2_value"] <= Select(v["arg1_range"], 0), False)) if n else
          If(v["arg2_value"] == Select(v["arg1_range"], 0), v["arg2_value"] <= Select(v["arg1_range"], 0), False))
)

def rule_1_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 1
        rule_1(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
