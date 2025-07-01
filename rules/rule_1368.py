import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 is of type string and its value is equal to either "sum", "mean", or "max", then for tensor v_2, the minimum element must be less than or equal to the maximum element. (Rule 1368)

rule_1368 = lambda s, v, n=False: (
    s.add(Not(If((Or(Or(v["arg1_value"] == 8, v["arg1_value"] == 7), v["arg1_value"] == 9)), Select(v["arg2_range"], 0) <= Select(v["arg2_range"], 1), False)) if n else
          If((Or(Or(v["arg1_value"] == 8, v["arg1_value"] == 7), v["arg1_value"] == 9)), Select(v["arg2_range"], 0) <= Select(v["arg2_range"], 1), False))
)

def rule_1368_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 1368
        rule_1368(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1368(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
