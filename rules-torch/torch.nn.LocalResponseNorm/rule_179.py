import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Alpha should be a reasonable finite value, not NaN or infinity (Rule 179)

rule_179 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] == v["arg1_value"], v["arg1_value"] < 1.0E7), v["arg1_value"] > -1.0E7)) if n else
          And(And(v["arg1_value"] == v["arg1_value"], v["arg1_value"] < 1.0E7), v["arg1_value"] > -1.0E7))
)

def rule_179_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 179
        rule_179(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_179(solver, {'arg1_value': arg1['value']}, neg)
