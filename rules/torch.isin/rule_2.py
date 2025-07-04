import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if assume_unique is True, elements and test_elements should not have duplicated values. (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, Or(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), Select(v["arg2_range"], 0) == Select(v["arg2_range"], 1)), False)) if n else
          If(v["arg3_value"] == True, Or(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), Select(v["arg2_range"], 0) == Select(v["arg2_range"], 1)), False))
)

def rule_2_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 2
        rule_2(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)
