import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# For numerical stability, if a tensor has very large values and beta is also large, the threshold should also be large. (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 1) > 100000, v["arg2_value"] > 100), v["arg3_value"] > 1000, True)) if n else
          If(And(Select(v["arg1_range"], 1) > 100000, v["arg2_value"] > 100), v["arg3_value"] > 1000, True))
)

def rule_71_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 71
        rule_71(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
