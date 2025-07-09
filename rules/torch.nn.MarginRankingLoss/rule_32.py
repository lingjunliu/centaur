import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If non-scalar input, must be 1D tensors with same size. Also, target must be int/float and input1/input2 must have same dtype (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(Or((And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0)), (And(And(And(And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), v["arg3_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0)), ((And(1 <= v["arg3_dtype"], v["arg3_dtype"] <= 8)))), (v["arg1_dtype"] == v["arg2_dtype"]))))) if n else
          Or((And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0)), (And(And(And(And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), v["arg3_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0)), ((And(1 <= v["arg3_dtype"], v["arg3_dtype"] <= 8)))), (v["arg1_dtype"] == v["arg2_dtype"])))))
)

def rule_32_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 32
        rule_32(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape']}, neg)
