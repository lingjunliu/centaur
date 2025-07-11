import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If n is specified then input tensor needs to be a complex type and the dim should be valid, or input tensor needs to be a real type and dim should be valid (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(Or((And(And(And(And(v["arg1_value"] > 0, v["arg2_dtype"] > 9), v["arg2_dtype"] < 12), v["arg3_value"] >= (0 - v["arg2_ndim"])), v["arg3_value"] < v["arg2_ndim"])), (And(And(And(And(v["arg1_value"] > 0, v["arg2_dtype"] > 6), v["arg2_dtype"] < 10), v["arg3_value"] >= (0 - v["arg2_ndim"])), v["arg3_value"] < v["arg2_ndim"])))) if n else
          Or((And(And(And(And(v["arg1_value"] > 0, v["arg2_dtype"] > 9), v["arg2_dtype"] < 12), v["arg3_value"] >= (0 - v["arg2_ndim"])), v["arg3_value"] < v["arg2_ndim"])), (And(And(And(And(v["arg1_value"] > 0, v["arg2_dtype"] > 6), v["arg2_dtype"] < 10), v["arg3_value"] >= (0 - v["arg2_ndim"])), v["arg3_value"] < v["arg2_ndim"]))))
)

def rule_82_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 82
        rule_82(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
