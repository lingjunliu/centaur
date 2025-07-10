import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Theta should have a floating-point dtype and if size is of length 4, then theta shape should be Nx2x3 else if size is of length 5, then theta shape should be Nx3x4. (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), (If(v["arg2_length"] == 4, And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 1) == 2), Select(v["arg1_shape"], 2) == 3), If(v["arg2_length"] == 5, And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 1) == 3), Select(v["arg1_shape"], 2) == 4), False))))) if n else
          And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), (If(v["arg2_length"] == 4, And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 1) == 2), Select(v["arg1_shape"], 2) == 3), If(v["arg2_length"] == 5, And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 1) == 3), Select(v["arg1_shape"], 2) == 4), False)))))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 43
        rule_43(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
