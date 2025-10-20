import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If y has dtype as integer, then its values should not be zero to avoid divide by zero errors. (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), And(Select(v["arg2_range"], 0) != 0, Select(v["arg2_range"], 1) != 0), True)) if n else
          If(Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), And(Select(v["arg2_range"], 0) != 0, Select(v["arg2_range"], 1) != 0), True))
)

def rule_67_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 67
        rule_67(solver, {'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
