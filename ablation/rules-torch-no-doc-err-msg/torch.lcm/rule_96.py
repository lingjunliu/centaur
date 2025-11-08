import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the first input is a tensor, the second is an integer, the second must be positive or zero, lower than 100, greater than -100, greater than the minimum value of first tensor, and less than the maximum value of the tensor (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, And(And(And(And(v["arg2_value"] >= 0, v["arg2_value"] < 100), v["arg2_value"] > -100), v["arg2_value"] > Select(v["arg1_range"], 0)), v["arg2_value"] < Select(v["arg1_range"], 1)), True)) if n else
          If(v["arg1_ndim"] > 0, And(And(And(And(v["arg2_value"] >= 0, v["arg2_value"] < 100), v["arg2_value"] > -100), v["arg2_value"] > Select(v["arg1_range"], 0)), v["arg2_value"] < Select(v["arg1_range"], 1)), True))
)

def rule_96_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 96
        rule_96(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
