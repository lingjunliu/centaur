import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If upscale factor is valid then the channel dimension must be a non-zero multiple of upscale factor squared (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] > 0, v["arg2_value"] < 100000), Or([And(i < (100 + 1), Select(v["arg1_shape"], v["arg1_ndim"] - 3) == i * (v["arg2_value"] * v["arg2_value"])) for i in range(6)]), True)) if n else
          If(And(v["arg2_value"] > 0, v["arg2_value"] < 100000), Or([And(i < (100 + 1), Select(v["arg1_shape"], v["arg1_ndim"] - 3) == i * (v["arg2_value"] * v["arg2_value"])) for i in range(6)]), True))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 35
        rule_35(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
