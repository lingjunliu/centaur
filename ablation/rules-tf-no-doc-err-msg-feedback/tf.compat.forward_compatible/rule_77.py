import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Both v_1 and v_2 should be valid dtypes for tensors and should be equal to each other or string v_3 equals "none" (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11), v["arg1_value"] == 12)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 1), v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12))), (Or(v["arg1_value"] == v["arg2_value"], v["arg3_value"] == 6)))) if n else
          And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11), v["arg1_value"] == 12)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 1), v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12))), (Or(v["arg1_value"] == v["arg2_value"], v["arg3_value"] == 6))))
)

def rule_77_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 77
        rule_77(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
