import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Final pixel_unshuffle validation (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_value"] > 0, (If(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 2) % v["arg2_value"] == 0, True))), (Or(Or(v["arg1_ndim"] >= 3, (And(And(v["arg1_ndim"] >= 0, Select(v["arg1_shape"], -1) > 0), Select(v["arg1_shape"], -2) > 0))), v["arg1_ndim"] < 2)))) if n else
          And(And(v["arg2_value"] > 0, (If(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 2) % v["arg2_value"] == 0, True))), (Or(Or(v["arg1_ndim"] >= 3, (And(And(v["arg1_ndim"] >= 0, Select(v["arg1_shape"], -1) > 0), Select(v["arg1_shape"], -2) > 0))), v["arg1_ndim"] < 2))))
)

def rule_49_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 49
        rule_49(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
