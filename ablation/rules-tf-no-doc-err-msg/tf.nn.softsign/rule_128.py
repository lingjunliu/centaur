import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Tuple length must be smaller than number of dimensions of tensor and tensor must have at least one dimension if dtype is int32 and tuple length is not 0. (Rule 128)

rule_128 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 3, v["arg2_length"] > 0), And(v["arg1_ndim"] > v["arg2_length"], v["arg1_ndim"] >= 1), True)) if n else
          If(And(v["arg1_dtype"] == 3, v["arg2_length"] > 0), And(v["arg1_ndim"] > v["arg2_length"], v["arg1_ndim"] >= 1), True))
)

def rule_128_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 128
        rule_128(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_128(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length']}, neg)
