import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When seed is not provided, lower bound and upper bound should be reasonably centered around 1 (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 0, And((v["arg1_value"] + v["arg2_value"]) / 2 > 0.5, (v["arg1_value"] + v["arg2_value"]) / 2 < 1.5), True)) if n else
          If(v["arg3_value"] == 0, And((v["arg1_value"] + v["arg2_value"]) / 2 > 0.5, (v["arg1_value"] + v["arg2_value"]) / 2 < 1.5), True))
)

def rule_43_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 43
        rule_43(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
