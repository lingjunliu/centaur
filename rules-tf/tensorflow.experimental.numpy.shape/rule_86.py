import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the shape[0] is greater than zero, the tuple should not contain duplicate value (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > 0, (And([Implies(i < (v["arg2_length"] - 2 + 1), Select(v["arg2_values"], i) != Select(v["arg2_values"], i + 1)) for i in range(6)])), False)) if n else
          If(Select(v["arg1_shape"], 0) > 0, (And([Implies(i < (v["arg2_length"] - 2 + 1), Select(v["arg2_values"], i) != Select(v["arg2_values"], i + 1)) for i in range(6)])), False))
)

def rule_86_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 86
        rule_86(solver, {'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
