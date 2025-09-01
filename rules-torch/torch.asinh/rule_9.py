import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The output tensor's dtype should be able to represent the asinh of the input tensor's values without overflow. (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 1, And(Select(v["arg1_range"], 0) > -0.79, Select(v["arg1_range"], 1) < 0.79), If(v["arg2_dtype"] == 2, And(Select(v["arg1_range"], 0) > -7.62E4, Select(v["arg1_range"], 1) < 7.62E4), If(v["arg2_dtype"] == 3, And(Select(v["arg1_range"], 0) > -5.24E8, Select(v["arg1_range"], 1) < 5.24E8), If(v["arg2_dtype"] == 4, And(Select(v["arg1_range"], 0) > -2.14E9, Select(v["arg1_range"], 1) < 2.14E9), If(v["arg2_dtype"] == 5, And(Select(v["arg1_range"], 0) > -9.22E18, Select(v["arg1_range"], 1) < 9.22E18), True)))))) if n else
          If(v["arg2_dtype"] == 1, And(Select(v["arg1_range"], 0) > -0.79, Select(v["arg1_range"], 1) < 0.79), If(v["arg2_dtype"] == 2, And(Select(v["arg1_range"], 0) > -7.62E4, Select(v["arg1_range"], 1) < 7.62E4), If(v["arg2_dtype"] == 3, And(Select(v["arg1_range"], 0) > -5.24E8, Select(v["arg1_range"], 1) < 5.24E8), If(v["arg2_dtype"] == 4, And(Select(v["arg1_range"], 0) > -2.14E9, Select(v["arg1_range"], 1) < 2.14E9), If(v["arg2_dtype"] == 5, And(Select(v["arg1_range"], 0) > -9.22E18, Select(v["arg1_range"], 1) < 9.22E18), True))))))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 9
        rule_9(solver, {'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)
