import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# seed must be an integer, and remapping for negative values preserves its integer nature (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 0, (v["arg1_value"] + 18446744073709551616) == ((v["arg1_value"] + 18446744073709551616) / 1) * 1, v["arg1_value"] == (v["arg1_value"] / 1) * 1)) if n else
          If(v["arg1_value"] < 0, (v["arg1_value"] + 18446744073709551616) == ((v["arg1_value"] + 18446744073709551616) / 1) * 1, v["arg1_value"] == (v["arg1_value"] / 1) * 1))
)

def rule_15_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 15
        rule_15(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_value': arg1['value']}, neg)
