import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# d_model must be divisible by nhead, and both must be positive. And d_model should be greater than 16. (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] % v["arg2_value"] == 0, v["arg1_value"] > 0), v["arg2_value"] > 0), v["arg1_value"] > 16)) if n else
          And(And(And(v["arg1_value"] % v["arg2_value"] == 0, v["arg1_value"] > 0), v["arg2_value"] > 0), v["arg1_value"] > 16))
)

def rule_76_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 76
        rule_76(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
