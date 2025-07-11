import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If preserve_aspect_ratio is true, and the original width > height, then the new width will be size[1] and the new height will be appropriately scaled. (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, (If(v["arg1_ndim"] == 4, (If(Select(v["arg1_shape"], 2) > Select(v["arg1_shape"], 1), True, False)), (If(Select(v["arg1_shape"], 1) > Select(v["arg1_shape"], 0), True, False)))), False)) if n else
          If(v["arg2_value"] == True, (If(v["arg1_ndim"] == 4, (If(Select(v["arg1_shape"], 2) > Select(v["arg1_shape"], 1), True, False)), (If(Select(v["arg1_shape"], 1) > Select(v["arg1_shape"], 0), True, False)))), False))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 34
        rule_34(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
