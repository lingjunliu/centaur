import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the increment is a float value, autocast should be enabled. and the increment should be larger than 0.1 (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] != 0, And(v["arg1_value"] == True, v["arg2_value"] > 0.1), False)) if n else
          If(v["arg2_value"] != 0, And(v["arg1_value"] == True, v["arg2_value"] > 0.1), False))
)

def rule_102_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 102
        rule_102(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
