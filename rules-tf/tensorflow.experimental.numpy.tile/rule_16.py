import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the number of dimensions in `a` is less than length of reps, dimensions of `a` are implicitly prepended with size one (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] < v["arg2_length"], And([Implies(i < (v["arg2_length"] - v["arg1_ndim"] - 1 + 1), Or(Select(v["arg1_shape"], i) == 1, Select(v["arg1_shape"], i) > 0)) for i in range(6)]), True)) if n else
          If(v["arg1_ndim"] < v["arg2_length"], And([Implies(i < (v["arg2_length"] - v["arg1_ndim"] - 1 + 1), Or(Select(v["arg1_shape"], i) == 1, Select(v["arg1_shape"], i) > 0)) for i in range(6)]), True))
)

def rule_16_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 16
        rule_16(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
