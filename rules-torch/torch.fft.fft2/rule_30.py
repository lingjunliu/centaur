import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If out is given, its shape must be compatible to the input after transformation (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_length"] == 0, v["arg1_ndim"] == v["arg2_ndim"], And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg2_shape"], Select(v["arg4_values"], i)) == Select(v["arg3_values"], i)) for i in range(6)]))) if n else
          If(v["arg3_length"] == 0, v["arg1_ndim"] == v["arg2_ndim"], And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg2_shape"], Select(v["arg4_values"], i)) == Select(v["arg3_values"], i)) for i in range(6)])))
)

def rule_30_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 30
        rule_30(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_values': arg4_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_values': arg4['values']}, neg)
