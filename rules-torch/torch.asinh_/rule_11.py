import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If no explicit dtype is specified, the result type should be at least float32 if the input is not complex. (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] < 9, Or(v["arg1_dtype"] >= 7, v["arg1_dtype"] == 6), False)) if n else
          If(v["arg1_dtype"] < 9, Or(v["arg1_dtype"] >= 7, v["arg1_dtype"] == 6), False))
)

def rule_11_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 11
        rule_11(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_dtype': arg1['dtype']}, neg)
