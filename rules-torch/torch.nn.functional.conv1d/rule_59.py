import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input and Weight must have the same dtype, all dimensions of tensor positive (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] == v["arg2_dtype"], (If(Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == Select(v["arg2_shape"], 0)) for i in range(6)]), True, Select(v["arg3_shape"], 0) == Select(v["arg2_shape"], 0))))) if n else
          And(v["arg1_dtype"] == v["arg2_dtype"], (If(Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == Select(v["arg2_shape"], 0)) for i in range(6)]), True, Select(v["arg3_shape"], 0) == Select(v["arg2_shape"], 0)))))
)

def rule_59_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 59
        rule_59(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)
