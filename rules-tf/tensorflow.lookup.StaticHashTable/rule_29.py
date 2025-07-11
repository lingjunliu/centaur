import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# default value must be within a reasonable range based on the value tensor (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_range"], 0) - 100 <= v["arg2_value"], v["arg2_value"] <= Select(v["arg1_range"], 1) + 100)) if n else
          And(Select(v["arg1_range"], 0) - 100 <= v["arg2_value"], v["arg2_value"] <= Select(v["arg1_range"], 1) + 100))
)

def rule_29_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 29
        rule_29(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
