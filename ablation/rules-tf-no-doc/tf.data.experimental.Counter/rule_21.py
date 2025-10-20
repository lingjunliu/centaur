import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# v_1 is a valid TF DType, either as a direct dtype object or its integer index in the TF dtype list (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 12, True, And(v["arg1_value"] >= 0, v["arg1_value"] < 11))) if n else
          If(v["arg1_value"] == 12, True, And(v["arg1_value"] >= 0, v["arg1_value"] < 11)))
)

def rule_21_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value']}, neg)
