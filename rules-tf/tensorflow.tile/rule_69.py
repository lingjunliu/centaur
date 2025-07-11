import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If all values of corresponding multiple dimension are 1, the shape of the output will have same values (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0), False)) if n else
          If(And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0), False))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 69
        rule_69(solver, {'arg1_range': arg1_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_range': arg1['range'], 'arg2_shape': arg2['shape']}, neg)
