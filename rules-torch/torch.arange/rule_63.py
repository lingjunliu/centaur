import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Step sign should align with difference between end and start (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(v["arg3_value"] > 0, v["arg1_value"] < v["arg2_value"])), (And(v["arg3_value"] < 0, v["arg1_value"] > v["arg2_value"]))), (And(v["arg3_value"] == 0, v["arg1_value"] == v["arg2_value"])))) if n else
          Or(Or((And(v["arg3_value"] > 0, v["arg1_value"] < v["arg2_value"])), (And(v["arg3_value"] < 0, v["arg1_value"] > v["arg2_value"]))), (And(v["arg3_value"] == 0, v["arg1_value"] == v["arg2_value"]))))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
