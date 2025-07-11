import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Output tensor type has to at least int32 if the output is specified, and input is int8, int16 or uint8 (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, If(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg2_dtype"] >= 3, False), False)) if n else
          If(v["arg2_ndim"] > 0, If(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg2_dtype"] >= 3, False), False))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 78
        rule_78(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
