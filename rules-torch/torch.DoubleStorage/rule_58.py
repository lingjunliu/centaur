import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Float size, when translated to bytes, should be non-negative (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 0, 0 - v["arg1_value"] * 8 <= 9223372036854775807, v["arg1_value"] * 8 <= 9223372036854775807)) if n else
          If(v["arg1_value"] < 0, 0 - v["arg1_value"] * 8 <= 9223372036854775807, v["arg1_value"] * 8 <= 9223372036854775807))
)

def rule_58_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value']}, neg)
