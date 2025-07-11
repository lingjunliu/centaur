import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if align_corners is true the size must be more than 1 (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(Select(v["arg2_shape"], 0) > 1, Select(v["arg2_shape"], 1) > 1), Or(v["arg1_value"] == False, v["arg1_value"] == True))) if n else
          If(v["arg1_value"] == True, And(Select(v["arg2_shape"], 0) > 1, Select(v["arg2_shape"], 1) > 1), Or(v["arg1_value"] == False, v["arg1_value"] == True)))
)

def rule_80_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)
