import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# output tensor must have the correct shape with keepdim = True - when tuple (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == True, (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Or((Or([And(k < (v["arg3_length"] - 1 + 1), i == Select(v["arg3_values"], k)) for k in range(6)])), (Select(v["arg2_shape"], i) == Select(v["arg1_shape"], i)))) for i in range(6)])), True)) if n else
          If(v["arg4_value"] == True, (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Or((Or([And(k < (v["arg3_length"] - 1 + 1), i == Select(v["arg3_values"], k)) for k in range(6)])), (Select(v["arg2_shape"], i) == Select(v["arg1_shape"], i)))) for i in range(6)])), True))
)

def rule_98_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == arg4)

        # Constraints for rule 98
        rule_98(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg3_length': arg3_length, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length'], 'arg4_value': arg4['value']}, neg)
