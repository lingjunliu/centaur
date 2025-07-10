import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Reduction is not constant and dtypes match if margin and p are non negative and at least the shapes are not zero (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(And(And(And(v["arg1_value"] != 20, v["arg2_dtype"] == v["arg3_dtype"]), v["arg3_dtype"] == v["arg4_dtype"]), v["arg5_value"] >= 0), v["arg6_value"] >= 0), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), (Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), (Or([And(i < (v["arg4_ndim"] - 1 + 1), Select(v["arg4_shape"], i) > 0) for i in range(6)]))), True, False)) if n else
          If(And(And(And(And(And(And(And(v["arg1_value"] != 20, v["arg2_dtype"] == v["arg3_dtype"]), v["arg3_dtype"] == v["arg4_dtype"]), v["arg5_value"] >= 0), v["arg6_value"] >= 0), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), (Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), (Or([And(i < (v["arg4_ndim"] - 1 + 1), Select(v["arg4_shape"], i) > 0) for i in range(6)]))), True, False))
)

def rule_99_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')
        arg5_value = Real('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 99
        rule_99(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
