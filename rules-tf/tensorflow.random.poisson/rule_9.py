import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dtype must be one of float16, float32, float64, int32 or int64, the corresponding integer representation can be any of 3, 4, 6, 7, 8. (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8)) if n else
          Or(Or(Or(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8))
)

def rule_9_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value']}, neg)
