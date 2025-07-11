import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The sum of padding should be equal to the shape of other tensor (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1) == Select(v["arg2_shape"], 0)) if n else
          Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1) == Select(v["arg2_shape"], 0))
)

def rule_50_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 50
        rule_50(solver, {'arg1_values': arg1_values, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_values': arg1['values'], 'arg2_shape': arg2['shape']}, neg)
