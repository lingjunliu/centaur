import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The desired number of samples is invalid and thus, will short circuit. When the v_1 and v_4 are set the output dimensions always has to equal zero. 1D and 2D decodes will also return zero. This means every condition has to exist. (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_shape"], 0) == 0, v["arg2_value"] == 0), True, True)) if n else
          If(And(Select(v["arg1_shape"], 0) == 0, v["arg2_value"] == 0), True, True))
)

def rule_108_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 108
        rule_108(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
