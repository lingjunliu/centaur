import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The input tensor must have dtype of float16, float32, float64, complex64, complex128 or dtype to avoid the mish_cpu error (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_dtype"] > 5, v["arg1_dtype"] < 11)), v["arg1_dtype"] == 12)) if n else
          Or((And(v["arg1_dtype"] > 5, v["arg1_dtype"] < 11)), v["arg1_dtype"] == 12))
)

def rule_65_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 65
        rule_65(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_dtype': arg1['dtype']}, neg)
