import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If out tensor is provided, its data type must be able to accommodate the result of sinh applied to the input tensor. (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 2, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 3, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 4, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 5, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 6, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 7, Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), If(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8, If(v["arg1_dtype"] == 9, Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 10, v["arg2_dtype"] == 10, False))))))))))) if n else
          If(v["arg1_dtype"] == 1, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 2, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 3, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 4, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 5, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 6, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 7, Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), If(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8, If(v["arg1_dtype"] == 9, Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 10, v["arg2_dtype"] == 10, False)))))))))))
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 2
        rule_2(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
