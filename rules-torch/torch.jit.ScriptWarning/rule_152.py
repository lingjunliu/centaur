import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# warning when we use sum on batch with smaller image size but the size are not a large number and may create slower optimization codes and implementation. (Rule 152)

rule_152 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg1_value"] == 8), (Select(v["arg2_shape"], 0) < 32)), (Select(v["arg2_shape"], 1) < 32)), (v["arg2_ndim"] == 4))) if n else
          And(And(And((v["arg1_value"] == 8), (Select(v["arg2_shape"], 0) < 32)), (Select(v["arg2_shape"], 1) < 32)), (v["arg2_ndim"] == 4)))
)

def rule_152_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 152
        rule_152(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_152(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
