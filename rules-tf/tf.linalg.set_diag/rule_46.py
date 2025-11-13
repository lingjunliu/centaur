import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If k is a tuple and input has at least 1 dimension, then k[0] must be a valid dimension (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, And(Select(v["arg1_values"], 0) >= (0 - v["arg2_ndim"]), Select(v["arg1_values"], 0) < v["arg2_ndim"]), True)) if n else
          If(v["arg2_ndim"] > 0, And(Select(v["arg1_values"], 0) >= (0 - v["arg2_ndim"]), Select(v["arg1_values"], 0) < v["arg2_ndim"]), True))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 46
        rule_46(solver, {'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim']}, neg)
