import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor must be of dimension 3,4, or 5 and if groups is given, dimension 0 must be divisible by the group size (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or((v["arg1_ndim"] == 3), (v["arg1_ndim"] == 4)), (v["arg1_ndim"] == 5))), (Or((v["arg2_value"] <= 1), (Select(v["arg1_shape"], 0) % v["arg2_value"] == 0))))) if n else
          And((Or(Or((v["arg1_ndim"] == 3), (v["arg1_ndim"] == 4)), (v["arg1_ndim"] == 5))), (Or((v["arg2_value"] <= 1), (Select(v["arg1_shape"], 0) % v["arg2_value"] == 0)))))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
