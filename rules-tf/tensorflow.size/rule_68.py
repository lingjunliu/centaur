import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype is int32 or int64, and tensor exists, ensure the shape is valid for the given dtype (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg2_value"] == 3, v["arg2_value"] == 4)), (And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), Select(v["arg1_shape"], i) >= 0) for i in range(6)])), True)) if n else
          If((Or(v["arg2_value"] == 3, v["arg2_value"] == 4)), (And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), Select(v["arg1_shape"], i) >= 0) for i in range(6)])), True))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 68
        rule_68(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
