import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 is equal to 0, then the result of the difference between tensor v_2 maximum shape and the multiplication of integer v_3 with two must be greater than zero (Rule 622)

rule_622 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 0, v["arg2_ndim"] > 0), (Select(v["arg2_range"], 1) - (v["arg3_value"] * 2)) > 0, False)) if n else
          If(And(v["arg1_value"] == 0, v["arg2_ndim"] > 0), (Select(v["arg2_range"], 1) - (v["arg3_value"] * 2)) > 0, False))
)

def rule_622_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 622
        rule_622(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_622(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
