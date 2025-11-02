import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If min and max of v1 are the same value, and the list of v2 is empty, return true (Rule 126)

rule_126 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), v["arg2_length"] == 0, True)) if n else
          If(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), v["arg2_length"] == 0, True))
)

def rule_126_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 126
        rule_126(solver, {'arg1_range': arg1_range, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_range': arg1['range'], 'arg2_length': arg2['length']}, neg)
