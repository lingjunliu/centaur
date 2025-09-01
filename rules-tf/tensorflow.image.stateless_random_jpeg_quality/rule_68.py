import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Quality and seed constraints (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(0 <= v["arg1_value"], v["arg1_value"] <= 100), 0 <= v["arg2_value"]), v["arg2_value"] <= 100), v["arg1_value"] < v["arg2_value"]), (v["arg1_value"] % 1) == 0), (v["arg2_value"] % 1) == 0), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) == 2), (Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4)))) if n else
          And(And(And(And(And(And(And(And(And(0 <= v["arg1_value"], v["arg1_value"] <= 100), 0 <= v["arg2_value"]), v["arg2_value"] <= 100), v["arg1_value"] < v["arg2_value"]), (v["arg1_value"] % 1) == 0), (v["arg2_value"] % 1) == 0), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) == 2), (Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4))))
)

def rule_68_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape']}, neg)
