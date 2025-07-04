import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If values is a scalar, then its dtype must be the same as sorted_sequence (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_dtype"] == 7, Or(v["arg2_value"] == 0.0, v["arg1_dtype"] == 4)), v["arg2_value"] == 0)) if n else
          And(And(v["arg1_dtype"] == 7, Or(v["arg2_value"] == 0.0, v["arg1_dtype"] == 4)), v["arg2_value"] == 0))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (float, np.floating)) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 15
        rule_15(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
