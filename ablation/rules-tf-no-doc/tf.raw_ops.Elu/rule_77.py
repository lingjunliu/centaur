import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the minimum of the tensor is a positive number, alpha must not have a value that is larger than a certain number  (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) > 0.0, v["arg2_value"] < 1000000.0, True)) if n else
          If(Select(v["arg1_range"], 0) > 0.0, v["arg2_value"] < 1000000.0, True))
)

def rule_77_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 77
        rule_77(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
