import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# lambd must be smaller than a threshold to avoid overflow for float16 tensors. If input tensor is complex64 or complex128, lambd should be smaller than the min/max of the real part of the tensor. Otherwise, it should be non-negative (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 6, v["arg1_value"] < 100, If(Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), And(v["arg1_value"] < Select(v["arg2_range"], 0), v["arg1_value"] < Select(v["arg2_range"], 1)), v["arg1_value"] >= 0))) if n else
          If(v["arg2_dtype"] == 6, v["arg1_value"] < 100, If(Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), And(v["arg1_value"] < Select(v["arg2_range"], 0), v["arg1_value"] < Select(v["arg2_range"], 1)), v["arg1_value"] >= 0)))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
