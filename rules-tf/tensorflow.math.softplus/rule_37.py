import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The max of the feature must be a number not a string (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_range"], 1) != 12, Select(v["arg1_range"], 1) != 13), Select(v["arg1_range"], 1) != 14), Select(v["arg1_range"], 1) != 15), Select(v["arg1_range"], 1) != 16), Select(v["arg1_range"], 1) != 17), Select(v["arg1_range"], 1) != 18), Select(v["arg1_range"], 1) != 19), Select(v["arg1_range"], 1) != 20), Select(v["arg1_range"], 1) != 21), Select(v["arg1_range"], 1) != 22)) if n else
          And(And(And(And(And(And(And(And(And(And(Select(v["arg1_range"], 1) != 12, Select(v["arg1_range"], 1) != 13), Select(v["arg1_range"], 1) != 14), Select(v["arg1_range"], 1) != 15), Select(v["arg1_range"], 1) != 16), Select(v["arg1_range"], 1) != 17), Select(v["arg1_range"], 1) != 18), Select(v["arg1_range"], 1) != 19), Select(v["arg1_range"], 1) != 20), Select(v["arg1_range"], 1) != 21), Select(v["arg1_range"], 1) != 22))
)

def rule_37_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 37
        rule_37(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_range': arg1['range']}, neg)
