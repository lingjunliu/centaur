import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If y is integer and any element of x is not an integer, then y should be non negative (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), If(Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), And(v["arg1_dtype"] != 3, v["arg1_dtype"] != 4)) for i in range(6)]), Select(v["arg2_range"], 0) >= 0, True), True)) if n else
          If(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), If(Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), And(v["arg1_dtype"] != 3, v["arg1_dtype"] != 4)) for i in range(6)]), Select(v["arg2_range"], 0) >= 0, True), True))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 42
        rule_42(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
