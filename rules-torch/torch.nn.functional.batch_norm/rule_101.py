import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input is 4D or higher, then running_mean, running_var, weight and bias should be 1D with size equal to input's number of features and the input should be float type (Rule 101)

rule_101 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 3, And(And(And(And(And(And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], 1)), (Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8))), (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))), (Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8))), (Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8))), True)) if n else
          If(v["arg1_ndim"] > 3, And(And(And(And(And(And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], 1)), (Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8))), (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))), (Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8))), (Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8))), True))
)

def rule_101_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

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
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))

        # Constraints for rule 101
        rule_101(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_101(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype']}, neg)
