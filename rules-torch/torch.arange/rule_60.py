import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# start, end, and step are consistent; if one is float, all must be (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), And(And((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), False)) if n else
          If(Or(Or((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), And(And((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), False))
)

def rule_60_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
