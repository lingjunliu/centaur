import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The tensor dimension should be equal to the length of a tuple only if they're more than 0. (Rule 126)

rule_126 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_ndim"] > 0), (v["arg2_length"] > 0)), v["arg1_ndim"] == v["arg2_length"], True)) if n else
          If(And((v["arg1_ndim"] > 0), (v["arg2_length"] > 0)), v["arg1_ndim"] == v["arg2_length"], True))
)

def rule_126_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 126
        rule_126(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
