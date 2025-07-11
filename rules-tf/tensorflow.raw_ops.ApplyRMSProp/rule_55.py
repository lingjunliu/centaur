import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# lr, rho, epsilon and momentum must be scalar and non-negative tensors of the same type. (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), Select(v["arg1_range"], 0) >= 0), Select(v["arg2_range"], 0) >= 0), Select(v["arg3_range"], 0) >= 0), Select(v["arg4_range"], 0) >= 0), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"])) if n else
          And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), Select(v["arg1_range"], 0) >= 0), Select(v["arg2_range"], 0) >= 0), Select(v["arg3_range"], 0) >= 0), Select(v["arg4_range"], 0) >= 0), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]))
)

def rule_55_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_dtype = Int('arg4_dtype')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 55
        rule_55(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim, 'arg4_dtype': arg4_dtype, 'arg4_range': arg4_range, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim'], 'arg4_dtype': arg4['dtype'], 'arg4_range': arg4['range'], 'arg4_ndim': arg4['ndim']}, neg)
