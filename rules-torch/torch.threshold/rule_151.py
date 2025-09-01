import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# value must be smaller then the threshold + 100. Also dtype must not be bool and ndim(v1 (Rule 151)

rule_151 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] < v["arg2_value"] + 100, v["arg3_dtype"] != 0), v["arg3_ndim"] == 2), Select(v["arg3_range"], 1) != Select(v["arg3_range"], 0))) if n else
          And(And(And(v["arg1_value"] < v["arg2_value"] + 100, v["arg3_dtype"] != 0), v["arg3_ndim"] == 2), Select(v["arg3_range"], 1) != Select(v["arg3_range"], 0)))
)

def rule_151_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 151
        rule_151(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_151(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
