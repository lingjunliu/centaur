import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Verify all parameters are valid (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If((And(And(v["arg1_ndim"] >= 3, (Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3))), v["arg1_dtype"] == 5)), True, And(False, If((And(And(And(And(And(And(0 <= v["arg2_value"], v["arg2_value"] <= 100), 0 <= v["arg3_value"]), v["arg3_value"] <= 100), v["arg2_value"] < v["arg3_value"]), (v["arg2_value"] % 1) == 0), (v["arg3_value"] % 1) == 0)), True, And(False, If((And(And(v["arg4_ndim"] == 1, Select(v["arg4_shape"], 0) == 2), (Or(v["arg4_dtype"] == 3, v["arg4_dtype"] == 4)))), True, False)))))) if n else
          If((And(And(v["arg1_ndim"] >= 3, (Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3))), v["arg1_dtype"] == 5)), True, And(False, If((And(And(And(And(And(And(0 <= v["arg2_value"], v["arg2_value"] <= 100), 0 <= v["arg3_value"]), v["arg3_value"] <= 100), v["arg2_value"] < v["arg3_value"]), (v["arg2_value"] % 1) == 0), (v["arg3_value"] % 1) == 0)), True, And(False, If((And(And(v["arg4_ndim"] == 1, Select(v["arg4_shape"], 0) == 2), (Or(v["arg4_dtype"] == 3, v["arg4_dtype"] == 4)))), True, False))))))
)

def rule_64_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 64
        rule_64(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype']}, neg)
