import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is bool and v_2 is tensor, number of dimensions of v_2 must be limited (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], v["arg2_ndim"] < 5, v["arg2_ndim"] < 10)) if n else
          If(v["arg1_value"], v["arg2_ndim"] < 5, v["arg2_ndim"] < 10))
)

def rule_72_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
