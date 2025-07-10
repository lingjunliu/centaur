import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the ngram width is a list, the length of ngram width must be less than the size of the last dimension in data (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, v["arg1_length"] <= Select(v["arg2_shape"], v["arg2_ndim"] - 1), False)) if n else
          If(v["arg2_ndim"] > 0, v["arg1_length"] <= Select(v["arg2_shape"], v["arg2_ndim"] - 1), False))
)

def rule_40_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 40
        rule_40(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
