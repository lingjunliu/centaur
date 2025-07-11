import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If channels is 1,3 or 4, output should be ndim >=3, and last dim should be 1,3,4 accordingly (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 3), v["arg1_value"] == 4), And(v["arg2_ndim"] >= 3, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == v["arg1_value"]), False)) if n else
          If(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 3), v["arg1_value"] == 4), And(v["arg2_ndim"] >= 3, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == v["arg1_value"]), False))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
