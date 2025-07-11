import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dummy rule that checks the dtype of contents combined with the value of channels. (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, True, v["arg2_value"] == v["arg2_value"])) if n else
          If(v["arg1_dtype"] == 11, True, v["arg2_value"] == v["arg2_value"]))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 45
        rule_45(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
