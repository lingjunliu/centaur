import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If align_corners is true, then the spatial dimensions of input and grid should correspond correctly (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == True, v["arg1_ndim"] == 4), And(And(And(Select(v["arg1_shape"], 2) > 0, Select(v["arg1_shape"], 3) > 0), Select(v["arg2_shape"], 1) > 0), Select(v["arg2_shape"], 2) > 0), If(And(v["arg3_value"] == True, v["arg1_ndim"] == 5), And(And(And(And(And(Select(v["arg1_shape"], 2) > 0, Select(v["arg1_shape"], 3) > 0), Select(v["arg1_shape"], 4) > 0), Select(v["arg2_shape"], 1) > 0), Select(v["arg2_shape"], 2) > 0), Select(v["arg2_shape"], 3) > 0), True))) if n else
          If(And(v["arg3_value"] == True, v["arg1_ndim"] == 4), And(And(And(Select(v["arg1_shape"], 2) > 0, Select(v["arg1_shape"], 3) > 0), Select(v["arg2_shape"], 1) > 0), Select(v["arg2_shape"], 2) > 0), If(And(v["arg3_value"] == True, v["arg1_ndim"] == 5), And(And(And(And(And(Select(v["arg1_shape"], 2) > 0, Select(v["arg1_shape"], 3) > 0), Select(v["arg1_shape"], 4) > 0), Select(v["arg2_shape"], 1) > 0), Select(v["arg2_shape"], 2) > 0), Select(v["arg2_shape"], 3) > 0), True)))
)

def rule_50_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 50
        rule_50(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
