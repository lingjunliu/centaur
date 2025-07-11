import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input is not complex, output is a zero-filled tensor with the same shape. (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), (And(v["arg1_ndim"] == v["arg2_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg2_range"], 0) == 0), Select(v["arg2_range"], 1) == 0)) for i in range(6)]))), False)) if n else
          If(And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), (And(v["arg1_ndim"] == v["arg2_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg2_range"], 0) == 0), Select(v["arg2_range"], 1) == 0)) for i in range(6)]))), False))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 21
        rule_21(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
