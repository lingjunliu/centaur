import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# x and y must be quantized tensors and have the same shape (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or((And(v["arg1_dtype"] == 1, v["arg2_dtype"] == 1)), (And(v["arg1_dtype"] == 5, v["arg2_dtype"] == 5))), (And(v["arg1_dtype"] == 3, v["arg2_dtype"] == 3))), (And(v["arg1_dtype"] == 2, v["arg2_dtype"] == 2))), (And(v["arg1_dtype"] == 16, v["arg2_dtype"] == 16)))), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))) if n else
          And((Or(Or(Or(Or((And(v["arg1_dtype"] == 1, v["arg2_dtype"] == 1)), (And(v["arg1_dtype"] == 5, v["arg2_dtype"] == 5))), (And(v["arg1_dtype"] == 3, v["arg2_dtype"] == 3))), (And(v["arg1_dtype"] == 2, v["arg2_dtype"] == 2))), (And(v["arg1_dtype"] == 16, v["arg2_dtype"] == 16)))), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
