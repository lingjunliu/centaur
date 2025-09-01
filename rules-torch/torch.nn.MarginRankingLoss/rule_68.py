import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input tensors have at least one dimension and that dimension has valid size, then all shape along dimension 0 should be same (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0), v["arg3_ndim"] > 0), Select(v["arg1_shape"], 0) > 0), Select(v["arg2_shape"], 0) > 0), Select(v["arg3_shape"], 0) > 0), And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0)), True)) if n else
          If(And(And(And(And(And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0), v["arg3_ndim"] > 0), Select(v["arg1_shape"], 0) > 0), Select(v["arg2_shape"], 0) > 0), Select(v["arg3_shape"], 0) > 0), And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0)), True))
)

def rule_68_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 68
        rule_68(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
