import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, lr, l1, l2, grad must have the same shape (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg5_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg6_shape"], i))) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg5_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg6_shape"], i))) for i in range(6)]))
)

def rule_20_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())
        arg6_shape = Array('arg6_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])
        for i in range(arg6.ndim):
            arg6_shape = Store(arg6_shape, i, arg6.shape[i])

        # Constraints for rule 20
        rule_20(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape, 'arg5_shape': arg5_shape, 'arg6_shape': arg6_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape'], 'arg5_shape': arg5['shape'], 'arg6_shape': arg6['shape']}, neg)
