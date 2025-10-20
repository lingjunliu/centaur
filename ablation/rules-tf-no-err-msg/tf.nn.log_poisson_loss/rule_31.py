import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If log_input has more than 0 dimensions, and if compute_full_loss is true, then the minimum of the targets should be positive or zero (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] > 0, v["arg3_value"] == True), Select(v["arg1_range"], 0) >= 0, True)) if n else
          If(And(v["arg2_ndim"] > 0, v["arg3_value"] == True), Select(v["arg1_range"], 0) >= 0, True))
)

def rule_31_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Bool('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == arg3)

        # Constraints for rule 31
        rule_31(solver, {'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
