import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 and v_2 is float, then the absolute of v_1 minus v_2 can not be smaller than 1e-8 (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_value"] - v["arg2_value"]) > 0, (v["arg1_value"] - v["arg2_value"]) > 0.00000001, (v["arg2_value"] - v["arg1_value"]) > 0.00000001)) if n else
          If((v["arg1_value"] - v["arg2_value"]) > 0, (v["arg1_value"] - v["arg2_value"]) > 0.00000001, (v["arg2_value"] - v["arg1_value"]) > 0.00000001))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 116
        rule_116(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
