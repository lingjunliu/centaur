import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# sparse_values data type must be one of float16, float32, float64, complex64, complex128 when sparse_indices has non-zero entries (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > 0, Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), False)) if n else
          If(Select(v["arg1_shape"], 0) > 0, Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), False))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 25
        rule_25(solver, {'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype']}, neg)
