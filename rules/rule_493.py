import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if tensor v_1 has ndim greater than 0, then tensor v_1 must satisfy that maximum element is greater than minimum element by at least a positive number v_2 (Rule 493)

rule_493 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, (Select(v["arg1_range"], 1) + (0 - Select(v["arg1_range"], 0))) > v["arg2_value"], False)) if n else
          If(v["arg1_ndim"] > 0, (Select(v["arg1_range"], 1) + (0 - Select(v["arg1_range"], 0))) > v["arg2_value"], False))
)

def rule_493_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 493
        rule_493(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_493(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
