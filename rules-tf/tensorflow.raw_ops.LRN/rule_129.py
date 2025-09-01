import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if depth_radius is high then bias must be less than 10 and greater than zero, and alpha must be within reasonable bounds (less than 100 (Rule 129)

rule_129 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 7, And(And(And(v["arg2_value"] < 10, v["arg2_value"] > 0), v["arg3_value"] < 100), v["arg3_value"] > 0), True)) if n else
          If(v["arg1_value"] > 7, And(And(And(v["arg2_value"] < 10, v["arg2_value"] > 0), v["arg3_value"] < 100), v["arg3_value"] > 0), True))
)

def rule_129_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 129
        rule_129(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_129(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
