import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Probability is within bounds, tensor dimension is acceptable, and tensor dtype is floating point (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(And(And((And(v["arg1_value"] >= 0.0, v["arg1_value"] <= 1.0)), (Or(v["arg2_ndim"] == 3, v["arg2_ndim"] == 4))), (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8)))) if n else
          And(And((And(v["arg1_value"] >= 0.0, v["arg1_value"] <= 1.0)), (Or(v["arg2_ndim"] == 3, v["arg2_ndim"] == 4))), (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8))))
)

def rule_33_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 33
        rule_33(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
