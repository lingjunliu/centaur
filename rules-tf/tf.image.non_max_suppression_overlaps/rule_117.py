import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Value range checks for numerical parameters (Rule 117)

rule_117 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(Select(v["arg1_range"], 0) > -10000.0, Select(v["arg1_range"], 1) < 10000.0), Select(v["arg2_range"], 0) > -10000.0), Select(v["arg2_range"], 1) < 10000.0), v["arg3_value"] >= 0), v["arg3_value"] <= 1), v["arg4_value"] > -10000.0), v["arg4_value"] < 10000.0)) if n else
          And(And(And(And(And(And(And(Select(v["arg1_range"], 0) > -10000.0, Select(v["arg1_range"], 1) < 10000.0), Select(v["arg2_range"], 0) > -10000.0), Select(v["arg2_range"], 1) < 10000.0), v["arg3_value"] >= 0), v["arg3_value"] <= 1), v["arg4_value"] > -10000.0), v["arg4_value"] < 10000.0))
)

def rule_117_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 117
        rule_117(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_117(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
