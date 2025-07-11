import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If out_type is integer, the string value must be a valid integer string. (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 3, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), If(v["arg2_value"] == 4, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), If(v["arg2_value"] == 5, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), If(v["arg2_value"] == 6, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), False))))) if n else
          If(v["arg2_value"] == 3, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), If(v["arg2_value"] == 4, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), If(v["arg2_value"] == 5, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), If(v["arg2_value"] == 6, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), True) for i in range(6)]), False)))))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 19
        rule_19(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
