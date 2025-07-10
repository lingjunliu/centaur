import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If depth_radius * 2 + 1 is greater than the feature map dimension of the tensor, then the result may be incorrect, but avoid crashing. (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 3) > 0, 2 * v["arg2_value"] + 1 <= Select(v["arg1_shape"], 3), False)) if n else
          If(Select(v["arg1_shape"], 3) > 0, 2 * v["arg2_value"] + 1 <= Select(v["arg1_shape"], 3), False))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 43
        rule_43(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
