import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dtype is one of the floating types and margin is positive. (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8)), v["arg2_value"] > 0)) if n else
          And((Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8)), v["arg2_value"] > 0))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 34
        rule_34(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
