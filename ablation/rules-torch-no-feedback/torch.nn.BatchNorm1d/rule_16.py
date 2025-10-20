import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If affine is true, then gamma and beta must be learnable parameters (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(v["arg2_ndim"] == 1, v["arg3_ndim"] == 1), True)) if n else
          If(v["arg1_value"] == True, And(v["arg2_ndim"] == 1, v["arg3_ndim"] == 1), True))
)

def rule_16_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 16
        rule_16(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim']}, neg)
