import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If training=True, then the standard deviation of the input should be greater than 0. This can be approximately checked by checking max(v_1 (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, Select(v["arg1_range"], 1) > Select(v["arg1_range"], 0), False)) if n else
          If(v["arg2_value"] == True, Select(v["arg1_range"], 1) > Select(v["arg1_range"], 0), False))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 21
        rule_21(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
