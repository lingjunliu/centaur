import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# V1 is less then dimension of v2 but more then 0. (Rule 120)

rule_120 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] < v["arg2_ndim"], v["arg1_value"] > 0)) if n else
          And(v["arg1_value"] < v["arg2_ndim"], v["arg1_value"] > 0))
)

def rule_120_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 120
        rule_120(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_120(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
