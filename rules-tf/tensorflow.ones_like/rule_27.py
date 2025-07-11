import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input tensor has at least one element, then dtype must be compatible (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 0, v["arg1_dtype"] == 0, If(v["arg2_value"] == 1, v["arg1_dtype"] == 1, If(v["arg2_value"] == 2, v["arg1_dtype"] == 2, If(v["arg2_value"] == 3, v["arg1_dtype"] == 3, If(v["arg2_value"] == 4, v["arg1_dtype"] == 4, If(v["arg2_value"] == 5, v["arg1_dtype"] == 5, If(v["arg2_value"] == 6, v["arg1_dtype"] == 6, If(v["arg2_value"] == 7, v["arg1_dtype"] == 7, If(v["arg2_value"] == 8, v["arg1_dtype"] == 8, If(v["arg2_value"] == 9, v["arg1_dtype"] == 9, If(v["arg2_value"] == 10, v["arg1_dtype"] == 10, If(v["arg2_value"] == 11, v["arg1_dtype"] == 11, False)))))))))))), False)) if n else
          If((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 0, v["arg1_dtype"] == 0, If(v["arg2_value"] == 1, v["arg1_dtype"] == 1, If(v["arg2_value"] == 2, v["arg1_dtype"] == 2, If(v["arg2_value"] == 3, v["arg1_dtype"] == 3, If(v["arg2_value"] == 4, v["arg1_dtype"] == 4, If(v["arg2_value"] == 5, v["arg1_dtype"] == 5, If(v["arg2_value"] == 6, v["arg1_dtype"] == 6, If(v["arg2_value"] == 7, v["arg1_dtype"] == 7, If(v["arg2_value"] == 8, v["arg1_dtype"] == 8, If(v["arg2_value"] == 9, v["arg1_dtype"] == 9, If(v["arg2_value"] == 10, v["arg1_dtype"] == 10, If(v["arg2_value"] == 11, v["arg1_dtype"] == 11, False)))))))))))), False))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
