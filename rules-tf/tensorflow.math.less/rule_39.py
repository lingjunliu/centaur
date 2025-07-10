import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if max of v1 equals the min of v1, and if v2 is integer, then the two tensors' datatype should be same (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 1) == Select(v["arg1_range"], 0), (Or(Or(Or(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), v["arg2_dtype"] == 2), v["arg2_dtype"] == 1), v["arg2_dtype"] == 5))), v["arg1_dtype"] == v["arg2_dtype"], False)) if n else
          If(And(Select(v["arg1_range"], 1) == Select(v["arg1_range"], 0), (Or(Or(Or(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), v["arg2_dtype"] == 2), v["arg2_dtype"] == 1), v["arg2_dtype"] == 5))), v["arg1_dtype"] == v["arg2_dtype"], False))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 39
        rule_39(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
