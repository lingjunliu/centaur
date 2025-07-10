import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# ksize and strides have the same length if the input tensor is 4D (Rule 263)

rule_263 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_ndim"] == 4, v["arg1_length"] == v["arg2_length"], False)) if n else
          If(v["arg3_ndim"] == 4, v["arg1_length"] == v["arg2_length"], False))
)

def rule_263_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 263
        rule_263(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_263(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim']}, neg)
