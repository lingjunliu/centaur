import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# three-input contraction bn,anm,bm->ba: shape and dtype consistency (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(v["arg1_value"] == 5, v["arg2_ndim"] == 2), v["arg3_ndim"] == 3), v["arg4_ndim"] == 2), Select(v["arg2_shape"], 0) == Select(v["arg4_shape"], 0)), Select(v["arg2_shape"], 1) == Select(v["arg3_shape"], 1)), Select(v["arg3_shape"], 2) == Select(v["arg4_shape"], 1)), v["arg2_dtype"] == v["arg3_dtype"]), v["arg2_dtype"] == v["arg4_dtype"])) if n else
          And(And(And(And(And(And(And(And(v["arg1_value"] == 5, v["arg2_ndim"] == 2), v["arg3_ndim"] == 3), v["arg4_ndim"] == 2), Select(v["arg2_shape"], 0) == Select(v["arg4_shape"], 0)), Select(v["arg2_shape"], 1) == Select(v["arg3_shape"], 1)), Select(v["arg3_shape"], 2) == Select(v["arg4_shape"], 1)), v["arg2_dtype"] == v["arg3_dtype"]), v["arg2_dtype"] == v["arg4_dtype"]))
)

def rule_8_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

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

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
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

        # Constraints for rule 8
        rule_8(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)
