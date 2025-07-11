import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All input tensors must have the same shape (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(And(And(And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg7_shape"], i))) for i in range(6)])), v["arg4_ndim"] == 0), v["arg5_ndim"] == 0), v["arg6_ndim"] == 0)) if n else
          And(And(And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg7_shape"], i))) for i in range(6)])), v["arg4_ndim"] == 0), v["arg5_ndim"] == 0), v["arg6_ndim"] == 0))
)

def rule_5_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

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
        if not isinstance(arg7, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg5_ndim = Int('arg5_ndim')
        arg6_ndim = Int('arg6_ndim')
        arg7_shape = Array('arg7_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg6_ndim == arg6.ndim)
        for i in range(arg7.ndim):
            arg7_shape = Store(arg7_shape, i, arg7.shape[i])

        # Constraints for rule 5
        rule_5(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_ndim': arg4_ndim, 'arg5_ndim': arg5_ndim, 'arg6_ndim': arg6_ndim, 'arg7_shape': arg7_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_ndim': arg4['ndim'], 'arg5_ndim': arg5['ndim'], 'arg6_ndim': arg6['ndim'], 'arg7_shape': arg7['shape']}, neg)
