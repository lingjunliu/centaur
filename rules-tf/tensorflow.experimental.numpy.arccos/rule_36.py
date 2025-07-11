import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype is complex, ensure that max and min absolute value are within bounds (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), And(Select(v["arg1_range"], 1) < v["arg2_value"], Select(v["arg1_range"], 0) > (0 - v["arg2_value"])), False)) if n else
          If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), And(Select(v["arg1_range"], 1) < v["arg2_value"], Select(v["arg1_range"], 0) > (0 - v["arg2_value"])), False))
)

def rule_36_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 36
        rule_36(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
