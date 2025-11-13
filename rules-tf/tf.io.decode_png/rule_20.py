import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If channels are 1, 3, or 4, then dtype should be either uint8 or uint16 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 3), v["arg1_value"] == 4), Or(v["arg2_value"] == 5, v["arg2_value"] == 2), True)) if n else
          If(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 3), v["arg1_value"] == 4), Or(v["arg2_value"] == 5, v["arg2_value"] == 2), True))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
