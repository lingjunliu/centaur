import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If seed and seed2 are provided, logits dtype must be float32 or float64 (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] != 0, v["arg2_value"] != 0), Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), False)) if n else
          If(Or(v["arg1_value"] != 0, v["arg2_value"] != 0), Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), False))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
