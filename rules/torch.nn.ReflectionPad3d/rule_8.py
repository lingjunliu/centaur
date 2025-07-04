import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the padding is a tuple with dimension length 4, then pad length should be less than or equal to two times input size for each dimension. (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] == 4, And(And(And(Select(v["arg2_values"], 0) <= 2 * Select(v["arg1_shape"], 2), Select(v["arg2_values"], 1) <= 2 * Select(v["arg1_shape"], 2)), Select(v["arg2_values"], 2) <= 2 * Select(v["arg1_shape"], 3)), Select(v["arg2_values"], 3) <= 2 * Select(v["arg1_shape"], 3)), False)) if n else
          If(v["arg2_length"] == 4, And(And(And(Select(v["arg2_values"], 0) <= 2 * Select(v["arg1_shape"], 2), Select(v["arg2_values"], 1) <= 2 * Select(v["arg1_shape"], 2)), Select(v["arg2_values"], 2) <= 2 * Select(v["arg1_shape"], 3)), Select(v["arg2_values"], 3) <= 2 * Select(v["arg1_shape"], 3)), False))
)

def rule_8_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 8
        rule_8(solver, {'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
