import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if input.dtype is half, CUDA support power of 2 or not, if not throw an error (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 7, And([Implies(i < (v["arg3_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 2, Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 4), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 8), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 16), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 32), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 64), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 128), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 256), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 512), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 1024), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 2048)) for i in range(6)]), True)) if n else
          If(v["arg1_dtype"] == 7, And([Implies(i < (v["arg3_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 2, Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 4), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 8), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 16), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 32), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 64), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 128), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 256), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 512), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 1024), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == 2048)) for i in range(6)]), True))
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
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 47
        rule_47(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
