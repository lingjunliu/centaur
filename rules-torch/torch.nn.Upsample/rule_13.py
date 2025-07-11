import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# scale_factor has to match input size if it is a tuple, and input is a 3D tensor (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 3, v["arg1_length"] == 1, False)) if n else
          If(v["arg2_ndim"] == 3, v["arg1_length"] == 1, False))
)

def rule_13_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 13
        rule_13(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)
