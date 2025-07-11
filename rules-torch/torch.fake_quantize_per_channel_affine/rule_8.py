import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# scale must be a 1-D tensor and zero_point must be Int32, Float32 or Half (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == 1, (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 7)))) if n else
          And(v["arg1_ndim"] == 1, (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 7))))
)

def rule_8_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 8
        rule_8(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
