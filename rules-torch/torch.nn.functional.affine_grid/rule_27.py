import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Combining everything: size is valid and theta is floating point with valid shape. (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_length"] == 4, v["arg1_length"] == 5)), And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > 0, (Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8))), (If(v["arg1_length"] == 4, And(And(v["arg2_ndim"] == 3, Select(v["arg2_shape"], 1) == 2), Select(v["arg2_shape"], 2) == 3), If(v["arg1_length"] == 5, And(And(v["arg2_ndim"] == 3, Select(v["arg2_shape"], 1) == 3), Select(v["arg2_shape"], 2) == 4), False))))) for i in range(6)]))) if n else
          And((Or(v["arg1_length"] == 4, v["arg1_length"] == 5)), And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > 0, (Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8))), (If(v["arg1_length"] == 4, And(And(v["arg2_ndim"] == 3, Select(v["arg2_shape"], 1) == 2), Select(v["arg2_shape"], 2) == 3), If(v["arg1_length"] == 5, And(And(v["arg2_ndim"] == 3, Select(v["arg2_shape"], 1) == 3), Select(v["arg2_shape"], 2) == 4), False))))) for i in range(6)])))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 27
        rule_27(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
