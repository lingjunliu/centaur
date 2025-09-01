import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# indices and segment_ids must have compatible dtypes and same length (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))) if n else
          And(And((Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
