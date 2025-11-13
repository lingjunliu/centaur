import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# For 4D tensors the valid axis range is limited. For tensors with different dimension the valid range has to be less than 3 if the axis is a positive number, and must be more than -101 if the axis is negative (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 4, And(v["arg2_value"] >= -101, v["arg2_value"] <= 3), If(v["arg2_value"] >= 0, v["arg2_value"] < 3, v["arg2_value"] > -101))) if n else
          If(v["arg1_ndim"] == 4, And(v["arg2_value"] >= -101, v["arg2_value"] <= 3), If(v["arg2_value"] >= 0, v["arg2_value"] < 3, v["arg2_value"] > -101)))
)

def rule_65_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 65
        rule_65(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
