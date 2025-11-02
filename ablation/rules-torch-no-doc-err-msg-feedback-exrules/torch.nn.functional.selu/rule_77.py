import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If all tensor values are same, alpha and scale should be default (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), And(And(And(v["arg2_value"] > 1.66, v["arg2_value"] < 1.68), v["arg3_value"] > 1.04), v["arg3_value"] < 1.06), True)) if n else
          If(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), And(And(And(v["arg2_value"] > 1.66, v["arg2_value"] < 1.68), v["arg3_value"] > 1.04), v["arg3_value"] < 1.06), True))
)

def rule_77_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 77
        rule_77(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
