import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# warning if we have small data value type and string is not non which could be an issue. (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] != 6, (Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2)))) if n else
          And(v["arg1_value"] != 6, (Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2))))
)

def rule_138_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 138
        rule_138(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
