import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# When a specific padding is specified as a tuple, if the input is 1D, tuple.len must be 1 (Rule 1058)

rule_1058 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 1, v["arg2_length"] == 1, False)) if n else
          If(v["arg1_ndim"] == 1, v["arg2_length"] == 1, False))
)

def rule_1058_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 1058
        rule_1058(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1058(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
