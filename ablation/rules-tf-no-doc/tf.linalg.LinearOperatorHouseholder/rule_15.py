import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If we are passing a vector to initialize the operator, and another boolean v_2 is provided which should indicate whether the vector will be normalized. Then v_2 can only be true (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, v["arg2_value"] == True, True)) if n else
          If(v["arg1_ndim"] > 0, v["arg2_value"] == True, True))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 15
        rule_15(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
