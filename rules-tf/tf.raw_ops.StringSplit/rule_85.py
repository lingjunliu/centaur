import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# the input shape 0 must be greater than 0, if skip_empty is true or false. (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_shape"], 0) > 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False)))) if n else
          And(Select(v["arg1_shape"], 0) > 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False))))
)

def rule_85_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 85
        rule_85(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
