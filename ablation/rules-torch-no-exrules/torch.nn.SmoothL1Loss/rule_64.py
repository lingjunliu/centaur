import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If both tensors are not scalar, then shapes should be equal when reduction is none and their dimensions are same. (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg3_value"] == 6), (v["arg1_ndim"] > 0)), (v["arg2_ndim"] > 0)), (v["arg1_ndim"] == v["arg2_ndim"]), True)) if n else
          If(And(And((v["arg3_value"] == 6), (v["arg1_ndim"] > 0)), (v["arg2_ndim"] > 0)), (v["arg1_ndim"] == v["arg2_ndim"]), True))
)

def rule_64_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 64
        rule_64(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
