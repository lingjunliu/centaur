import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If incompatible_shape_error is false and x is empty, then incompatible_shape_error must be true (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == False, v["arg1_ndim"] == 0), v["arg2_value"] == True, False)) if n else
          If(And(v["arg2_value"] == False, v["arg1_ndim"] == 0), v["arg2_value"] == True, False))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)

        # Constraints for rule 57
        rule_57(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
