import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is float16, value must be representable as half-precision float without overflow (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, Or([And(p < (2048 + 1), v["arg2_value"] == p * 0.0009765625) for p in range(6)]), False)) if n else
          If(v["arg1_dtype"] == 6, Or([And(p < (2048 + 1), v["arg2_value"] == p * 0.0009765625) for p in range(6)]), False))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
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
