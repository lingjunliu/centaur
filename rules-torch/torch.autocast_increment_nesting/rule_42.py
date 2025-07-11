import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If increment is a float value between -1 and 1 and divisible by 0.5, it must be zero. (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(And(And(-1 < v["arg1_value"], v["arg1_value"] < 1), v["arg1_value"] / 0.5 == (v["arg1_value"] * 2)), v["arg1_value"] == 0, False)) if n else
          If(And(And(-1 < v["arg1_value"], v["arg1_value"] < 1), v["arg1_value"] / 0.5 == (v["arg1_value"] * 2)), v["arg1_value"] == 0, False))
)

def rule_42_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value']}, neg)
