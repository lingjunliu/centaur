import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If alpha is 1 and both tensors are integral, then promote the output - different than rule 4 in previous output (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 1, If(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 6), If(And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 6), True, False), False), False)) if n else
          If(v["arg3_value"] == 1, If(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 6), If(And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 6), True, False), False), False))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (float, np.floating)) or (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 29
        rule_29(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
