import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The minimum value for int types must always be less than max (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 1, -128 < 127, If(v["arg1_value"] == 2, -32768 < 32767, If(v["arg1_value"] == 3, -2147483648 < 2147483647, If(v["arg1_value"] == 4, -9223372036854775808 < 9223372036854775807, If(v["arg1_value"] == 5, 0 < 255, False)))))) if n else
          If(v["arg1_value"] == 1, -128 < 127, If(v["arg1_value"] == 2, -32768 < 32767, If(v["arg1_value"] == 3, -2147483648 < 2147483647, If(v["arg1_value"] == 4, -9223372036854775808 < 9223372036854775807, If(v["arg1_value"] == 5, 0 < 255, False))))))
)

def rule_18_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 18
        rule_18(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_value': arg1['value']}, neg)
