import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# indices and segment_ids must be 1 dimensional and of int32 or int64 type, indices must be less than the size of grad in dimension 0 (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), Select(v["arg2_range"], 1) < Select(v["arg3_shape"], 0))) if n else
          And(And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), Select(v["arg2_range"], 1) < Select(v["arg3_shape"], 0)))
)

def rule_44_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 44
        rule_44(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_shape': arg3['shape']}, neg)
