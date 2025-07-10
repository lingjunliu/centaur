import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# check dtype of tensor is float32 or float64. (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9)) if n else
          Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9))
)

def rule_79_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 79
        rule_79(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_dtype': arg1['dtype']}, neg)
