import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Ratio tests at multiple sample ranges (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] < 22050, (v["arg1_value"] + v["arg2_value"]) / (v["arg4_value"] + v["arg5_value"] + 0.0001) < 1, If(v["arg3_value"] > 44000, (v["arg1_value"] + v["arg2_value"]) / (v["arg4_value"] + v["arg5_value"] + 0.0001) < 2, False))) if n else
          If(v["arg3_value"] < 22050, (v["arg1_value"] + v["arg2_value"]) / (v["arg4_value"] + v["arg5_value"] + 0.0001) < 1, If(v["arg3_value"] > 44000, (v["arg1_value"] + v["arg2_value"]) / (v["arg4_value"] + v["arg5_value"] + 0.0001) < 2, False)))
)

def rule_68_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (float, np.floating)) or (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg4_value = Real('arg4_value')
        arg5_value = Real('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
