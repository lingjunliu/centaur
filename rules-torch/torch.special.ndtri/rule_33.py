import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor must have a dtype of float32, float64, complex64, complex128 and its values should lie within the range of 0 to 1. (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not((And(And((Or(Or(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 11)), (Select(v["arg1_range"], 0) >= 0.0)), (Select(v["arg1_range"], 1) <= 1.0)))) if n else
          (And(And((Or(Or(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 11)), (Select(v["arg1_range"], 0) >= 0.0)), (Select(v["arg1_range"], 1) <= 1.0))))
)

def rule_33_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 33
        rule_33(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)
