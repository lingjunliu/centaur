import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If alpha is specified, and the input tensor is integral, then alpha must be an integer expressible as a float (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 6)), (Or([And(x < (10 + 1), v["arg2_value"] == x) for x in range(6)])), False)) if n else
          If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 6)), (Or([And(x < (10 + 1), v["arg2_value"] == x) for x in range(6)])), False))
)

def rule_70_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 70
        rule_70(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
