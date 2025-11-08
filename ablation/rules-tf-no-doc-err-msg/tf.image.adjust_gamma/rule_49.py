import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Image values should be in range(v_4,v_5 (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] < v["arg3_value"], (And(Select(v["arg1_range"], 1) <= v["arg3_value"], Select(v["arg1_range"], 0) >= v["arg2_value"])), (And(Select(v["arg1_range"], 1) <= 1, Select(v["arg1_range"], 0) >= 0)))) if n else
          If(v["arg2_value"] < v["arg3_value"], (And(Select(v["arg1_range"], 1) <= v["arg3_value"], Select(v["arg1_range"], 0) >= v["arg2_value"])), (And(Select(v["arg1_range"], 1) <= 1, Select(v["arg1_range"], 0) >= 0))))
)

def rule_49_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 49
        rule_49(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
