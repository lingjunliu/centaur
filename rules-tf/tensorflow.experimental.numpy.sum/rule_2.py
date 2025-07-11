import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If keepdims is true, the output tensor should have the same number of dimensions as the input. (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, v["arg1_ndim"] == v["arg1_ndim"], False)) if n else
          If(v["arg2_value"] == True, v["arg1_ndim"] == v["arg1_ndim"], False))
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 2
        rule_2(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
