import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Prevent Storage size calculation overflowed, limit total elements (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] < 1, True, If(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) < 1000000000, If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 1000000000, If(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) < 1000000000, False))))) if n else
          If(v["arg1_ndim"] < 1, True, If(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) < 1000000000, If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 1000000000, If(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) < 1000000000, False)))))
)

def rule_91_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 91
        rule_91(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
