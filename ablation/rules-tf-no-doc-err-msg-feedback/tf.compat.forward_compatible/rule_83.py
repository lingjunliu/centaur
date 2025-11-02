import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tensor v_1's dtype must be in this union of types, and v_2 must be "channels_first" or "channels_last" (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11), v["arg1_value"] == 12)), (Or(v["arg2_value"] == 25, v["arg2_value"] == 24)))) if n else
          And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11), v["arg1_value"] == 12)), (Or(v["arg2_value"] == 25, v["arg2_value"] == 24))))
)

def rule_83_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 83
        rule_83(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
