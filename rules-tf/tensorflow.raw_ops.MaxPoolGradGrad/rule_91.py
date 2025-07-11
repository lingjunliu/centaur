import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input is 4D, Ksize and Strides must have elements larger than 0 (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 4, And((And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 0) for i in range(6)])), (And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]))), False)) if n else
          If(v["arg1_ndim"] == 4, And((And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 0) for i in range(6)])), (And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]))), False))
)

def rule_91_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 91
        rule_91(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
