import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the input tensor has the bfloat16 dtype, then the maximum value must be less than a certain value (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, Select(v["arg1_range"], 1) < 7.0e+36, False)) if n else
          If(v["arg1_dtype"] == 6, Select(v["arg1_range"], 1) < 7.0e+36, False))
)

def rule_64_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 64
        rule_64(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)
