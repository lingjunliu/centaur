import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the dtype is float16, the values should be in representable range to avoid overflow and underflow (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And((Select(v["arg1_range"], 0) > -65500), (Select(v["arg1_range"], 1) < 65500)), False)) if n else
          If(v["arg1_dtype"] == 6, And((Select(v["arg1_range"], 0) > -65500), (Select(v["arg1_range"], 1) < 65500)), False))
)

def rule_19_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 19
        rule_19(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
