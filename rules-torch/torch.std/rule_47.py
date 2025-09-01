import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if dim is a tuple and the output tensor is given, shape of output tensor is equal to the input except for the dimensions given in dim, where the shape is equal to 1 (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == v["arg3_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), If((Or([And(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], j) == i) for j in range(6)])), Select(v["arg3_shape"], i) == 1, Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i))) for i in range(6)]))) if n else
          And(v["arg1_ndim"] == v["arg3_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), If((Or([And(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], j) == i) for j in range(6)])), Select(v["arg3_shape"], i) == 1, Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i))) for i in range(6)])))
)

def rule_47_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 47
        rule_47(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
