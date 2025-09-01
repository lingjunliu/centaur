import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The number of threads should not be a prime if the tensor is 3D (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 3, And([Implies(i < (v["arg1_value"] - 1 + 1), v["arg1_value"] % i != 0) for i in range(6)]), True)) if n else
          If(v["arg2_ndim"] == 3, And([Implies(i < (v["arg1_value"] - 1 + 1), v["arg1_value"] % i != 0) for i in range(6)]), True))
)

def rule_54_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 54
        rule_54(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
