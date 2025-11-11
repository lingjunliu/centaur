import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dtype must be string or integer type only: codes 1–5 for ints or 11 for string (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(Or((And(1 <= v["arg1_value"], v["arg1_value"] <= 5)), v["arg1_value"] == 11)) if n else
          Or((And(1 <= v["arg1_value"], v["arg1_value"] <= 5)), v["arg1_value"] == 11))
)

def rule_18_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 18
        rule_18(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_value': arg1['value']}, neg)
