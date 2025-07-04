import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If deterministic algorithms are enabled, the output is related to the shape and dtype of the tensor (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg2_shape"], 0) > 0, v["arg2_dtype"] == 7), v["arg1_value"] == True, v["arg1_value"] == False)) if n else
          If(And(Select(v["arg2_shape"], 0) > 0, v["arg2_dtype"] == 7), v["arg1_value"] == True, v["arg1_value"] == False))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
