import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The dtype cannot be complex64 or complex128 (9 or 10 (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_value"] != 9), (v["arg1_value"] != 10))) if n else
          And((v["arg1_value"] != 9), (v["arg1_value"] != 10)))
)

def rule_14_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value']}, neg)
