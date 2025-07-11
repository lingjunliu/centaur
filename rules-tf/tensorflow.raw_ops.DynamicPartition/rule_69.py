import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the partition has more dimensions than the data, then the shapes of the partitions must be greater than 0 (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > v["arg1_ndim"], And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]), False)) if n else
          If(v["arg2_ndim"] > v["arg1_ndim"], And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]), False))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 69
        rule_69(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
