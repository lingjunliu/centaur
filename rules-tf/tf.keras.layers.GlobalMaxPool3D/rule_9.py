import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# spatial dimensions must be positive based on the data format (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 24, (And([Implies(i < (3 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (And([Implies(i < (4 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])))) if n else
          If(v["arg2_value"] == 24, (And([Implies(i < (3 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (And([Implies(i < (4 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 9
        rule_9(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
