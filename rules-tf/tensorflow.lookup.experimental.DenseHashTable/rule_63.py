import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# empty_key and deleted_key must have the same shape and dtype as key_dtype (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_dtype"] == v["arg1_value"], v["arg3_dtype"] == v["arg1_value"]), If(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0, And((And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == 1)), If(v["arg3_ndim"] == 0, v["arg2_ndim"] == 0, (And(v["arg3_ndim"] == 1, Select(v["arg3_shape"], 0) == 1))))))) if n else
          And(And(v["arg2_dtype"] == v["arg1_value"], v["arg3_dtype"] == v["arg1_value"]), If(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0, And((And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == 1)), If(v["arg3_ndim"] == 0, v["arg2_ndim"] == 0, (And(v["arg3_ndim"] == 1, Select(v["arg3_shape"], 0) == 1)))))))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
