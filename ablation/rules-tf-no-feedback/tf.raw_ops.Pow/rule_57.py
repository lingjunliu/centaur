import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x is an integer and some values are negatives, then some values in y cannot be a fraction  (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4)), Select(v["arg1_range"], 0) < 0), Or((Select(v["arg2_range"], 0) == 0), (Select(v["arg2_range"], 1) == 1)), True)) if n else
          If(And((Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4)), Select(v["arg1_range"], 0) < 0), Or((Select(v["arg2_range"], 0) == 0), (Select(v["arg2_range"], 1) == 1)), True))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 57
        rule_57(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_range': arg2['range']}, neg)
