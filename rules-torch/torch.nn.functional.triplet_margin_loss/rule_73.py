import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Margin and P must be non negative if Reduction is not constant and swap is Boolean (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] != 20, And(And(v["arg1_value"] >= 0, v["arg2_value"] >= 0), (Or(v["arg4_value"] == True, v["arg4_value"] == False))), False)) if n else
          If(v["arg3_value"] != 20, And(And(v["arg1_value"] >= 0, v["arg2_value"] >= 0), (Or(v["arg4_value"] == True, v["arg4_value"] == False))), False))
)

def rule_73_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_string_values.index(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 73
        rule_73(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
