import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

#  If padding is not used, the output is not padded. (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Select(v["arg2_shape"], i) % 4 != 0) for i in range(6)]), True)) if n else
          If(v["arg1_value"] == False, And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Select(v["arg2_shape"], i) % 4 != 0) for i in range(6)]), True))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 57
        rule_57(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)
