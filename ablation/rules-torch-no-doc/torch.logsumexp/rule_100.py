import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dim is a tuple and keepdim is true, out must have the same size for non reduced dimensions (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(Or([And(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], j) == i) for j in range(6)]), True, Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i))) for i in range(6)]), True)) if n else
          If(v["arg3_value"] == True, And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(Or([And(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], j) == i) for j in range(6)]), True, Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i))) for i in range(6)]), True))
)

def rule_100_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == arg3)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 100
        rule_100(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape']}, neg)
