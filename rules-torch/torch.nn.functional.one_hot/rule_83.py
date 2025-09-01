import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# class values must be smaller than num_classes, if num_classes is not -1 and long data type is required, and the tensor must have reasonable shape (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] != -1, v["arg1_dtype"] == 4), And(Select(v["arg1_range"], 1) < v["arg2_value"], Select(v["arg1_shape"], 0) < 1000), True)) if n else
          If(And(v["arg2_value"] != -1, v["arg1_dtype"] == 4), And(Select(v["arg1_range"], 1) < v["arg2_value"], Select(v["arg1_shape"], 0) < 1000), True))
)

def rule_83_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 83
        rule_83(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
