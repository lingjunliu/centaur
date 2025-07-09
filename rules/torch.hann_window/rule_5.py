import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# dtype must be one of the floating point types or None, represented by 0 if no dtype is given (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 0)) if n else
          Or(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 0))
)

def rule_5_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 5
        rule_5(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_value': arg1['value']}, neg)
