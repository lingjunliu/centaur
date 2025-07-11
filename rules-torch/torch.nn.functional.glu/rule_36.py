import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dimension is None, it will default to -1, otherwise, dimension should be within range and an integer (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 6, True, (If(And(And((v["arg2_value"] >= (0 - v["arg1_ndim"])), (v["arg2_value"] < v["arg1_ndim"])), (Or([And(i < (100 + 1), v["arg2_value"] == i) for i in range(6)]))), True, False)))) if n else
          If(v["arg2_value"] == 6, True, (If(And(And((v["arg2_value"] >= (0 - v["arg1_ndim"])), (v["arg2_value"] < v["arg1_ndim"])), (Or([And(i < (100 + 1), v["arg2_value"] == i) for i in range(6)]))), True, False))))
)

def rule_36_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 36
        rule_36(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
