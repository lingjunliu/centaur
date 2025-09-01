import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Padding as int should not be excessively large relative to other dimension sizes (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (2 + 1), If(Select(v["arg1_shape"], i + v["arg1_ndim"] - 3) < 100, v["arg2_value"] < 10000 / Select(v["arg1_shape"], (i + 1) % 3 + v["arg1_ndim"] - 3) / Select(v["arg1_shape"], (i + 2) % 3 + v["arg1_ndim"] - 3), True)) for i in range(6)])) if n else
          And([Implies(i < (2 + 1), If(Select(v["arg1_shape"], i + v["arg1_ndim"] - 3) < 100, v["arg2_value"] < 10000 / Select(v["arg1_shape"], (i + 1) % 3 + v["arg1_ndim"] - 3) / Select(v["arg1_shape"], (i + 2) % 3 + v["arg1_ndim"] - 3), True)) for i in range(6)]))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 39
        rule_39(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
