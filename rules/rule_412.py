import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If bool v_1 equals true, then the difference between max and min value of v_2 tensor must be larger than the constant 0.5 (Rule 412)

rule_412 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], (Select(v["arg2_range"], 1) + (0 - Select(v["arg2_range"], 0))) > 0.5, False)) if n else
          If(v["arg1_value"], (Select(v["arg2_range"], 1) + (0 - Select(v["arg2_range"], 0))) > 0.5, False))
)

def rule_412_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 412
        rule_412(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_412(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
