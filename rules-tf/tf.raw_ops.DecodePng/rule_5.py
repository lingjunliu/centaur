import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if channels is set to a specific value, the shape of the output tensor depends on it, but we cannot determine it statically (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 1, v["arg2_ndim"] >= 2, (If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), v["arg2_ndim"] >= 3, True)))) if n else
          If(v["arg1_value"] == 1, v["arg2_ndim"] >= 2, (If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), v["arg2_ndim"] >= 3, True))))
)

def rule_5_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 5
        rule_5(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
