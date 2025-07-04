import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if inplace is false, then the minimum absolute value of the tensor elements should be less than or equal to 3 (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == False, Or(Select(v["arg1_range"], 1) >= 3, Select(v["arg1_range"], 0) <= -3), False)) if n else
          If(v["arg2_value"] == False, Or(Select(v["arg1_range"], 1) >= 3, Select(v["arg1_range"], 0) <= -3), False))
)

def rule_17_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 17
        rule_17(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
