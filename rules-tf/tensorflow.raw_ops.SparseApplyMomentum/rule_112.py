import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var and accum must have compatible dtypes. If var is int, accum must be int. If var is float, accum must be float. If var is other type, accum must also have the same type as var. (Rule 112)

rule_112 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5)), (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 10))), v["arg1_dtype"] == 12), v["arg1_dtype"] == v["arg2_dtype"], False)) if n else
          If(Or(Or((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5)), (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 10))), v["arg1_dtype"] == 12), v["arg1_dtype"] == v["arg2_dtype"], False))
)

def rule_112_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 112
        rule_112(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_112(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
