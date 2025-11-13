import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If images is 4D, then size must not be None (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 4, v["arg3_value"] == False), And(Select(v["arg2_shape"], 0) > 0, Select(v["arg2_shape"], 1) > 0), True)) if n else
          If(And(v["arg1_ndim"] == 4, v["arg3_value"] == False), And(Select(v["arg2_shape"], 0) > 0, Select(v["arg2_shape"], 1) > 0), True))
)

def rule_22_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 22
        rule_22(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
