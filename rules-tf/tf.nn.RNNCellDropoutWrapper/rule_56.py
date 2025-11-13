import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dtype cannot be the torch.int64 DType, using exclusion in its range. (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 11, (And(v["arg1_value"] >= 0, v["arg1_value"] != 4)), (v["arg1_value"] == 12))) if n else
          If(v["arg1_value"] < 11, (And(v["arg1_value"] >= 0, v["arg1_value"] != 4)), (v["arg1_value"] == 12)))
)

def rule_56_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 56
        rule_56(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_value': arg1['value']}, neg)
