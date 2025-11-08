import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The minimum of a tensor must be less than the integer v1, but the maximum of the same tensor must be greater than the integer v1 (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg2_range"], 0) < v["arg1_value"], Select(v["arg2_range"], 1) > v["arg1_value"])) if n else
          And(Select(v["arg2_range"], 0) < v["arg1_value"], Select(v["arg2_range"], 1) > v["arg1_value"]))
)

def rule_95_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 95
        rule_95(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
