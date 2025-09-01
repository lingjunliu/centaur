import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Checks both the shapes are consistent with one another. (Rule 139)

rule_139 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg1_ndim"] >= 0), (Or(v["arg3_length"] == 0, v["arg1_ndim"] == v["arg3_length"]))), (Or(Select(v["arg1_shape"], 0) >= 0, Select(v["arg2_shape"], 0) >= 0)))) if n else
          And(And((v["arg1_ndim"] >= 0), (Or(v["arg3_length"] == 0, v["arg1_ndim"] == v["arg3_length"]))), (Or(Select(v["arg1_shape"], 0) >= 0, Select(v["arg2_shape"], 0) >= 0))))
)

def rule_139_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 139
        rule_139(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_139(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_length': arg3['length']}, neg)
