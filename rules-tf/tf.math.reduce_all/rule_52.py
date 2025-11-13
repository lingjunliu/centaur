import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# For a 4D tensor, the axis must be an integer between -101 and 3 inclusive. For other tensors the axis must be between -ndim and ndim -1, or it must be None which is represented as -1 (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 4, Or((And(v["arg2_value"] >= -101, v["arg2_value"] <= 3)), v["arg2_value"] == -1), And(v["arg2_value"] >= (0 - v["arg1_ndim"]), v["arg2_value"] <= (v["arg1_ndim"] - 1)))) if n else
          If(v["arg1_ndim"] == 4, Or((And(v["arg2_value"] >= -101, v["arg2_value"] <= 3)), v["arg2_value"] == -1), And(v["arg2_value"] >= (0 - v["arg1_ndim"]), v["arg2_value"] <= (v["arg1_ndim"] - 1))))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 52
        rule_52(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
