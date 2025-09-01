import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If use_locking is true, then dimensions of all tensors must be positive (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) > 0, And([Implies(i < (v["arg3_ndim"] - 1 + 1), And(Select(v["arg3_shape"], i) > 0, And([Implies(i < (v["arg4_ndim"] - 1 + 1), And(Select(v["arg4_shape"], i) > 0, And([Implies(i < (v["arg5_ndim"] - 1 + 1), And(Select(v["arg5_shape"], i) > 0, And([Implies(i < (v["arg6_ndim"] - 1 + 1), And(Select(v["arg6_shape"], i) > 0, And([Implies(i < (v["arg7_ndim"] - 1 + 1), And(Select(v["arg7_shape"], i) > 0, And([Implies(i < (v["arg8_ndim"] - 1 + 1), Select(v["arg8_shape"], i) > 0) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)])), True)) if n else
          If(v["arg1_value"] == True, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) > 0, And([Implies(i < (v["arg3_ndim"] - 1 + 1), And(Select(v["arg3_shape"], i) > 0, And([Implies(i < (v["arg4_ndim"] - 1 + 1), And(Select(v["arg4_shape"], i) > 0, And([Implies(i < (v["arg5_ndim"] - 1 + 1), And(Select(v["arg5_shape"], i) > 0, And([Implies(i < (v["arg6_ndim"] - 1 + 1), And(Select(v["arg6_shape"], i) > 0, And([Implies(i < (v["arg7_ndim"] - 1 + 1), And(Select(v["arg7_shape"], i) > 0, And([Implies(i < (v["arg8_ndim"] - 1 + 1), Select(v["arg8_shape"], i) > 0) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)])), True))
)

def rule_35_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
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
        if not isinstance(arg8, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())
        arg6_ndim = Int('arg6_ndim')
        arg6_shape = Array('arg6_shape', IntSort(), IntSort())
        arg7_ndim = Int('arg7_ndim')
        arg7_shape = Array('arg7_shape', IntSort(), IntSort())
        arg8_ndim = Int('arg8_ndim')
        arg8_shape = Array('arg8_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_ndim == arg5.ndim)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])
        solver.add(arg6_ndim == arg6.ndim)
        for i in range(arg6.ndim):
            arg6_shape = Store(arg6_shape, i, arg6.shape[i])
        solver.add(arg7_ndim == arg7.ndim)
        for i in range(arg7.ndim):
            arg7_shape = Store(arg7_shape, i, arg7.shape[i])
        solver.add(arg8_ndim == arg8.ndim)
        for i in range(arg8.ndim):
            arg8_shape = Store(arg8_shape, i, arg8.shape[i])

        # Constraints for rule 35
        rule_35(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim, 'arg5_shape': arg5_shape, 'arg5_ndim': arg5_ndim, 'arg6_shape': arg6_shape, 'arg6_ndim': arg6_ndim, 'arg7_shape': arg7_shape, 'arg7_ndim': arg7_ndim, 'arg8_shape': arg8_shape, 'arg8_ndim': arg8_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim'], 'arg5_shape': arg5['shape'], 'arg5_ndim': arg5['ndim'], 'arg6_shape': arg6['shape'], 'arg6_ndim': arg6['ndim'], 'arg7_shape': arg7['shape'], 'arg7_ndim': arg7['ndim'], 'arg8_shape': arg8['shape'], 'arg8_ndim': arg8['ndim']}, neg)
