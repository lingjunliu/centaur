import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor has non-negative values, or it is a complex dtype tensor (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(Or((Select(v["arg1_range"], 0) >= 0), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)))) if n else
          Or((Select(v["arg1_range"], 0) >= 0), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))))
)

def rule_40_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 40
        rule_40(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
