import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The absolute value of axis should be less than the number of dimensions of params, considering batch_dims and axis has to be int32 or int64 and batch_dims less than the ndim of params and indices (Rule 122)

rule_122 = lambda s, v, n=False: (
    s.add(Not(And(And((If(v["arg3_value"] < 0, 0 - v["arg3_value"] < v["arg1_ndim"] - v["arg4_value"], v["arg3_value"] < v["arg1_ndim"] - v["arg4_value"])), v["arg4_value"] >= 0), v["arg4_value"] <= v["arg2_ndim"])) if n else
          And(And((If(v["arg3_value"] < 0, 0 - v["arg3_value"] < v["arg1_ndim"] - v["arg4_value"], v["arg3_value"] < v["arg1_ndim"] - v["arg4_value"])), v["arg4_value"] >= 0), v["arg4_value"] <= v["arg2_ndim"]))
)

def rule_122_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 122
        rule_122(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_122(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
