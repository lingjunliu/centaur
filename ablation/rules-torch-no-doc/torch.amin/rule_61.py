import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if out is specified, the shape needs to be right (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == False, And((v["arg1_ndim"] - v["arg3_length"]) == v["arg2_ndim"], And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == Select(v["arg1_shape"], i)) for i in range(6)])), True)) if n else
          If(v["arg4_value"] == False, And((v["arg1_ndim"] - v["arg3_length"]) == v["arg2_ndim"], And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == Select(v["arg1_shape"], i)) for i in range(6)])), True))
)

def rule_61_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 61
        rule_61(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_length': arg3_length, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_length': arg3['length'], 'arg4_value': arg4['value']}, neg)
