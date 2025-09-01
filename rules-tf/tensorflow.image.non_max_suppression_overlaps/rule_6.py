import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# scores and overlap_threshold must be a valid float (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(And(And(And(Select(v["arg1_range"], 0) > -100000000.0, Select(v["arg1_range"], 1) < 100000000.0), v["arg2_value"] > -100000000.0), v["arg2_value"] < 100000000.0)) if n else
          And(And(And(Select(v["arg1_range"], 0) > -100000000.0, Select(v["arg1_range"], 1) < 100000000.0), v["arg2_value"] > -100000000.0), v["arg2_value"] < 100000000.0))
)

def rule_6_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 6
        rule_6(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
