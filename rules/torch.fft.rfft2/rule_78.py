import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If s and dim are given, the elements within them must be valid dimensions for the input, must have the same length, and elements must be unique. And they should only be non-empty only when input size >0 (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == 0) for i in range(6)])), And(v["arg2_length"] == 0, v["arg3_length"] == 0), And((v["arg2_length"] == v["arg3_length"]), If(v["arg2_length"] > 0, And((And([Implies(i < (v["arg2_length"] - 1 + 1), And((0 - v["arg1_ndim"]) <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) < v["arg1_ndim"])) for i in range(6)])), (And([Implies(i < (v["arg2_length"] - 1 + 1), And([Implies(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) != Select(v["arg2_values"], j)) for j in range(6)])) for i in range(6)]))), False)))) if n else
          If((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == 0) for i in range(6)])), And(v["arg2_length"] == 0, v["arg3_length"] == 0), And((v["arg2_length"] == v["arg3_length"]), If(v["arg2_length"] > 0, And((And([Implies(i < (v["arg2_length"] - 1 + 1), And((0 - v["arg1_ndim"]) <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) < v["arg1_ndim"])) for i in range(6)])), (And([Implies(i < (v["arg2_length"] - 1 + 1), And([Implies(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) != Select(v["arg2_values"], j)) for j in range(6)])) for i in range(6)]))), False))))
)

def rule_78_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 78
        rule_78(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_length': arg3['length']}, neg)
