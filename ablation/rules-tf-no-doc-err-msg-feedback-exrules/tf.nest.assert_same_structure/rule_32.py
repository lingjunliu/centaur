import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x is a tensor and y is a str, then str y must be included in supported string list. (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 1), v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12), v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 15), v["arg2_value"] == 16), v["arg2_value"] == 17), v["arg2_value"] == 18), v["arg2_value"] == 19), v["arg2_value"] == 20), v["arg2_value"] == 21), v["arg2_value"] == 22), v["arg2_value"] == 23), v["arg2_value"] == 24), v["arg2_value"] == 25)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 1), v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12), v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 15), v["arg2_value"] == 16), v["arg2_value"] == 17), v["arg2_value"] == 18), v["arg2_value"] == 19), v["arg2_value"] == 20), v["arg2_value"] == 21), v["arg2_value"] == 22), v["arg2_value"] == 23), v["arg2_value"] == 24), v["arg2_value"] == 25))
)

def rule_32_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 32
        rule_32(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg2_value': arg2['value']}, neg)
