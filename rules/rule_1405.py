import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if v_1 is "...ij->...ji" and v_2 is a tensor, then the maximum value in v_2 should not be greater than 1000 and the minimum value should not be less than -1000 (Rule 1405)

rule_1405 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 4, And(Select(v["arg2_range"], 1) <= 1000, Select(v["arg2_range"], 0) >= -1000), False)) if n else
          If(v["arg1_value"] == 4, And(Select(v["arg2_range"], 1) <= 1000, Select(v["arg2_range"], 0) >= -1000), False))
)

def rule_1405_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 1405
        rule_1405(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1405(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
