import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the product between max and min of tensor v_1 is less than 0, then dimension of v_1 must be greater than 0 and v_2 integer needs to be between -1 and 1 (Rule 448)

rule_448 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) * Select(v["arg1_range"], 1) < 0, And(And(v["arg1_ndim"] > 0, v["arg2_value"] >= -1), v["arg2_value"] <= 1), False)) if n else
          If(Select(v["arg1_range"], 0) * Select(v["arg1_range"], 1) < 0, And(And(v["arg1_ndim"] > 0, v["arg2_value"] >= -1), v["arg2_value"] <= 1), False))
)

def rule_448_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 448
        rule_448(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_448(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
