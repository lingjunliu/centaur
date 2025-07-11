import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If out is provided, the dtype must be double if x is double, and dimensions must align. Plus dim should be valid dimension. (Rule 126)

rule_126 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 8, And(And(And(v["arg2_dtype"] == 8, (If(v["arg3_value"] == -1, v["arg1_ndim"] == v["arg2_ndim"] + 1, False))), v["arg3_value"] >= (0 - v["arg1_ndim"])), v["arg3_value"] < v["arg1_ndim"]), False)) if n else
          If(v["arg1_dtype"] == 8, And(And(And(v["arg2_dtype"] == 8, (If(v["arg3_value"] == -1, v["arg1_ndim"] == v["arg2_ndim"] + 1, False))), v["arg3_value"] >= (0 - v["arg1_ndim"])), v["arg3_value"] < v["arg1_ndim"]), False))
)

def rule_126_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 126
        rule_126(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
