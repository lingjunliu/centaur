import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# recompute_scale_factor is True, scale_factor is specified, input tensor dimensions need to match scale_factor dimensions (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, v["arg2_length"] == v["arg3_ndim"] - 2, False)) if n else
          If(v["arg1_value"] == True, v["arg2_length"] == v["arg3_ndim"] - 2, False))
)

def rule_61_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 61
        rule_61(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim']}, neg)
