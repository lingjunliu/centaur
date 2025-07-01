import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 has the value tanh AND v_2 is a tensor then max(v_2 (Rule 1435)

rule_1435 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 11, And(Select(v["arg2_range"], 1) > Select(v["arg2_range"], 0), (And(Select(v["arg2_range"], 0) >= -1, Select(v["arg2_range"], 1) <= 1))), False)) if n else
          If(v["arg1_value"] == 11, And(Select(v["arg2_range"], 1) > Select(v["arg2_range"], 0), (And(Select(v["arg2_range"], 0) >= -1, Select(v["arg2_range"], 1) <= 1))), False))
)

def rule_1435_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 1435
        rule_1435(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1435(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
