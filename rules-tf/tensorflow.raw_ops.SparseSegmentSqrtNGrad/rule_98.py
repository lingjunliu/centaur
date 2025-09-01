import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# indices and segment_ids must satisfy that indices has the same shape as segment_ids, and both have dimension of 1 and data type of int32 or int64, also segment ids value must be greater or equal to 0 and less than output_dim0 (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), v["arg1_ndim"] == 1), v["arg2_ndim"] == 1), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) < Select(v["arg3_range"], 0))) if n else
          And(And(And(And(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), v["arg1_ndim"] == 1), v["arg2_ndim"] == 1), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) < Select(v["arg3_range"], 0)))
)

def rule_98_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 98
        rule_98(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_range': arg3['range']}, neg)
