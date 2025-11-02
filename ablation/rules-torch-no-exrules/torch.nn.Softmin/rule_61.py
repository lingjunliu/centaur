import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Preventing boolean and char tensor errors. Also checks for complex types. Using list and if. (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg1_dtype"] == 0, v["arg1_dtype"] == 5), v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), False, True)) if n else
          If(Or(Or(Or(v["arg1_dtype"] == 0, v["arg1_dtype"] == 5), v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), False, True))
)

def rule_61_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 61
        rule_61(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_dtype': arg1['dtype']}, neg)
