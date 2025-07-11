import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If endpoint is false and num is 1, then start and stop should be the same (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, v["arg2_value"] == 1), And(Select(v["arg3_range"], 0) == Select(v["arg4_range"], 0), Select(v["arg3_range"], 1) == Select(v["arg4_range"], 1)), False)) if n else
          If(And(v["arg1_value"] == False, v["arg2_value"] == 1), And(Select(v["arg3_range"], 0) == Select(v["arg4_range"], 0), Select(v["arg3_range"], 1) == Select(v["arg4_range"], 1)), False))
)

def rule_39_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range']}, neg)
