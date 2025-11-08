import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input is complex64, Tout must be float32, and If input is complex128, Tout must be float64, and otherwise, the rule is false (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And((v["arg1_dtype"] == 10), (v["arg2_value"] == 8))), (And((v["arg1_dtype"] == 11), (v["arg2_value"] == 9)))), (And(And((v["arg1_dtype"] != 10), (v["arg1_dtype"] != 11)), False)))) if n else
          Or(Or((And((v["arg1_dtype"] == 10), (v["arg2_value"] == 8))), (And((v["arg1_dtype"] == 11), (v["arg2_value"] == 9)))), (And(And((v["arg1_dtype"] != 10), (v["arg1_dtype"] != 11)), False))))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 25
        rule_25(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
