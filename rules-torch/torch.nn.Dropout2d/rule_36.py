import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If probability is invalid and ndim is valid then raise value error (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_value"] < 0.0, v["arg1_value"] > 1.0)), (Or(v["arg2_ndim"] == 3, v["arg2_ndim"] == 4))), Or([And(i < (0 + 1), False) for i in range(6)]), True)) if n else
          If(And((Or(v["arg1_value"] < 0.0, v["arg1_value"] > 1.0)), (Or(v["arg2_ndim"] == 3, v["arg2_ndim"] == 4))), Or([And(i < (0 + 1), False) for i in range(6)]), True))
)

def rule_36_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
