import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check that A's shape is valid given B's shape, using recursion and explicit multiplication only for up to 3 dimensions, avoiding overflow for larger tensors (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] >= v["arg2_ndim"], (If(v["arg2_ndim"] == 0, True, If(v["arg2_ndim"] == 1, Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], v["arg1_ndim"] - 1), If(v["arg2_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) == Select(v["arg1_shape"], v["arg1_ndim"] - 2) * Select(v["arg1_shape"], v["arg1_ndim"] - 1), If(v["arg2_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) == Select(v["arg1_shape"], v["arg1_ndim"] - 3) * Select(v["arg1_shape"], v["arg1_ndim"] - 2) * Select(v["arg1_shape"], v["arg1_ndim"] - 1), True))))), True)) if n else
          If(v["arg1_ndim"] >= v["arg2_ndim"], (If(v["arg2_ndim"] == 0, True, If(v["arg2_ndim"] == 1, Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], v["arg1_ndim"] - 1), If(v["arg2_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) == Select(v["arg1_shape"], v["arg1_ndim"] - 2) * Select(v["arg1_shape"], v["arg1_ndim"] - 1), If(v["arg2_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) == Select(v["arg1_shape"], v["arg1_ndim"] - 3) * Select(v["arg1_shape"], v["arg1_ndim"] - 2) * Select(v["arg1_shape"], v["arg1_ndim"] - 1), True))))), True))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 19
        rule_19(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim']}, neg)
