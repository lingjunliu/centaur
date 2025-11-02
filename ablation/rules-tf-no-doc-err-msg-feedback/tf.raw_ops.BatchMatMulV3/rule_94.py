import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ensure the rank of v1 and v2 are at least 2 if either adj_x or adj_y is true (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg3_value"] == True, v["arg4_value"] == True), And(v["arg1_ndim"] >= 2, v["arg2_ndim"] >= 2), True)) if n else
          If(Or(v["arg3_value"] == True, v["arg4_value"] == True), And(v["arg1_ndim"] >= 2, v["arg2_ndim"] >= 2), True))
)

def rule_94_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 94
        rule_94(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
