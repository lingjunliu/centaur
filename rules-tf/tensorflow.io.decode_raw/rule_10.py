import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# fixed_length has to be greater than or equal to the size of out_type (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, v["arg1_value"] >= (If(v["arg2_value"] == 0, 1, If(v["arg2_value"] == 1, 1, If(v["arg2_value"] == 2, 2, If(v["arg2_value"] == 3, 4, If(v["arg2_value"] == 4, 8, If(v["arg2_value"] == 5, 1, If(v["arg2_value"] == 6, 2, If(v["arg2_value"] == 7, 4, If(v["arg2_value"] == 8, 8, If(v["arg2_value"] == 9, 8, If(v["arg2_value"] == 10, 16, 1)))))))))))), False)) if n else
          If(v["arg1_value"] > 0, v["arg1_value"] >= (If(v["arg2_value"] == 0, 1, If(v["arg2_value"] == 1, 1, If(v["arg2_value"] == 2, 2, If(v["arg2_value"] == 3, 4, If(v["arg2_value"] == 4, 8, If(v["arg2_value"] == 5, 1, If(v["arg2_value"] == 6, 2, If(v["arg2_value"] == 7, 4, If(v["arg2_value"] == 8, 8, If(v["arg2_value"] == 9, 8, If(v["arg2_value"] == 10, 16, 1)))))))))))), False))
)

def rule_10_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
