import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# DType must either be a TF DType object, or an integer less than 11, identifying one (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_value"] >= 0, v["arg1_value"] < 11)), (v["arg1_value"] == 12))) if n else
          Or((And(v["arg1_value"] >= 0, v["arg1_value"] < 11)), (v["arg1_value"] == 12)))
)

def rule_30_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value']}, neg)
