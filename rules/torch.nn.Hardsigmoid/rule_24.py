import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If inplace is set to false, then at least one element must be within the range [-3,3] (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == False, Or([And(x < (Select(v["arg1_range"], 1) + 1), And(x >= -3, x <= 3)) for x in range(6)]), False)) if n else
          If(v["arg2_value"] == False, Or([And(x < (Select(v["arg1_range"], 1) + 1), And(x >= -3, x <= 3)) for x in range(6)]), False))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 24
        rule_24(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
