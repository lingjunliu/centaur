import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If axis is None and keepdims is true, then the rank is maintained as 1. (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == -1, v["arg1_value"] == True), v["arg3_ndim"] == 1, False)) if n else
          If(And(v["arg2_value"] == -1, v["arg1_value"] == True), v["arg3_ndim"] == 1, False))
)

def rule_60_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
