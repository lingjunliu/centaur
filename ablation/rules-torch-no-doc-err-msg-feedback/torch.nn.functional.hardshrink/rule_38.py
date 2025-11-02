import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the tensor contains both positive and negative numbers, lambda should be chosen such that it falls between them (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) > 0), And(Select(v["arg1_range"], 0) <= v["arg2_value"], v["arg2_value"] <= Select(v["arg1_range"], 1)), True)) if n else
          If(And(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) > 0), And(Select(v["arg1_range"], 0) <= v["arg2_value"], v["arg2_value"] <= Select(v["arg1_range"], 1)), True))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 38
        rule_38(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
