import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If indices and grad are not empty, and indices' dtype is int64 then var should have dtype of float64 or int64 or complex128 (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(If(And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg2_shape"], 0) > 0), v["arg1_dtype"] == 4), (Or(Or(v["arg3_dtype"] == 8, v["arg3_dtype"] == 4), v["arg3_dtype"] == 10)), False)) if n else
          If(And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg2_shape"], 0) > 0), v["arg1_dtype"] == 4), (Or(Or(v["arg3_dtype"] == 8, v["arg3_dtype"] == 4), v["arg3_dtype"] == 10)), False))
)

def rule_33_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 33
        rule_33(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype']}, neg)
