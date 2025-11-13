import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Keys should not have any duplicate values if key_dtype is not string (Rule 127)

rule_127 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 11, True, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], i) != Select(v["arg1_shape"], j)) for j in range(6)])) for i in range(6)]))) if n else
          If(v["arg2_value"] == 11, True, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], i) != Select(v["arg1_shape"], j)) for j in range(6)])) for i in range(6)])))
)

def rule_127_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 127
        rule_127(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_127(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
