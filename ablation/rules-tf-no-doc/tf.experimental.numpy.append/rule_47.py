import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the inputs are 1D tensors, the resulting array is also 1D tensor (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == -1, True, If(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), v["arg1_ndim"] == 1, True))) if n else
          If(v["arg3_value"] == -1, True, If(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), v["arg1_ndim"] == 1, True)))
)

def rule_47_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 47
        rule_47(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
