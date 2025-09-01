import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Num_lower must be within valid range or negative and num_upper must be within valid range or negative, separate conditions (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(And((And((v["arg1_ndim"] >= 2), (Or((v["arg2_value"] < 0), (v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 2)))))), Or((And((v["arg1_ndim"] >= 2), (Or((v["arg3_value"] < 0), (v["arg3_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)))))), (And(And((v["arg1_ndim"] < 2), (v["arg2_value"] < 0)), (v["arg3_value"] < 0)))))) if n else
          And((And((v["arg1_ndim"] >= 2), (Or((v["arg2_value"] < 0), (v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 2)))))), Or((And((v["arg1_ndim"] >= 2), (Or((v["arg3_value"] < 0), (v["arg3_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)))))), (And(And((v["arg1_ndim"] < 2), (v["arg2_value"] < 0)), (v["arg3_value"] < 0))))))
)

def rule_30_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 30
        rule_30(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
