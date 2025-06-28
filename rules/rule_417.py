import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If float v_1 is positive, and the max value in v_2 tensor is less than 100, then string v_3 must not equal to "tanh" (Rule 417)

rule_417 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, Select(v["arg2_range"], 1) < 100), v["arg3_value"] != 11, False)) if n else
          If(And(v["arg1_value"] > 0, Select(v["arg2_range"], 1) < 100), v["arg3_value"] != 11, False))
)

def rule_417_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == list_of_string_values.index(arg3))

        # Constraints for rule 417
        rule_417(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_417(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)
