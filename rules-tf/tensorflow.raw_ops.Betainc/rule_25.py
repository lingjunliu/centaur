import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if any dimension of b is 0, then the corresponding dimensions of a and x must be 0 (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg2_ndim"] - 1 + 1), If(Select(v["arg2_shape"], i) == 0, And(Select(v["arg1_shape"], i) == 0, Select(v["arg3_shape"], i) == 0), True)) for i in range(6)])) if n else
          And([Implies(i < (v["arg2_ndim"] - 1 + 1), If(Select(v["arg2_shape"], i) == 0, And(Select(v["arg1_shape"], i) == 0, Select(v["arg3_shape"], i) == 0), True)) for i in range(6)]))
)

def rule_25_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 25
        rule_25(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape']}, neg)
