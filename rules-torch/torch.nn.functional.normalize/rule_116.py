import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If the input tensor is complex then norm `p` cannot be infinity or zero (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(And(10 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), And(v["arg2_value"] < 1000000000, v["arg2_value"] > 0), False)) if n else
          If(And(10 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), And(v["arg2_value"] < 1000000000, v["arg2_value"] > 0), False))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 116
        rule_116(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
