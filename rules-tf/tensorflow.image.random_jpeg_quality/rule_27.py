import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# image properties and quality and seed and value (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And((And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11), v["arg1_dtype"] != 12)), v["arg1_ndim"] == 3), (Or(Select(v["arg1_shape"], 2) == 1, Select(v["arg1_shape"], 2) == 3))), v["arg2_value"] >= 0), v["arg2_value"] <= 100), v["arg3_value"] >= 0), v["arg3_value"] <= 100), v["arg2_value"] < v["arg3_value"]), v["arg4_value"] >= 0), Select(v["arg1_range"], 1) <= 255), Select(v["arg1_range"], 0) >= 0)) if n else
          And(And(And(And(And(And(And(And(And(And((And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11), v["arg1_dtype"] != 12)), v["arg1_ndim"] == 3), (Or(Select(v["arg1_shape"], 2) == 1, Select(v["arg1_shape"], 2) == 3))), v["arg2_value"] >= 0), v["arg2_value"] <= 100), v["arg3_value"] >= 0), v["arg3_value"] <= 100), v["arg2_value"] < v["arg3_value"]), v["arg4_value"] >= 0), Select(v["arg1_range"], 1) <= 255), Select(v["arg1_range"], 0) >= 0))
)

def rule_27_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 27
        rule_27(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
