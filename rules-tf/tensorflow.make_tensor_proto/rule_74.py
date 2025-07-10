import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The product of the shape dimensions must equal the number of elements in the tensor (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] > 0, (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), False)) if n else
          If(v["arg2_length"] > 0, (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), False))
)

def rule_74_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 74
        rule_74(solver, {'arg1_shape': arg1_shape, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_shape': arg1['shape'], 'arg2_length': arg2['length']}, neg)
