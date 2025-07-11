import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Sufficient conditions to avoid errors with fake_quantize_per_channel_affine - emphasis on necessary checks (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg1_dtype"] == 8), (v["arg1_ndim"] == 1)), (Or(Or((v["arg2_dtype"] == 3), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)))), (v["arg2_ndim"] == 1))) if n else
          And(And(And((v["arg1_dtype"] == 8), (v["arg1_ndim"] == 1)), (Or(Or((v["arg2_dtype"] == 3), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)))), (v["arg2_ndim"] == 1)))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
