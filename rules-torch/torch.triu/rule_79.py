import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input tensor and the output tensor are both present but they have different dtypes, and the input tensor's dtype is a float, then a RuntimeError will occur about dtype mismatch. (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), (v["arg1_dtype"] != v["arg2_dtype"])), (v["arg2_ndim"] > 0))) if n else
          And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), (v["arg1_dtype"] != v["arg2_dtype"])), (v["arg2_ndim"] > 0)))
)

def rule_79_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 79
        rule_79(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
