import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Check if the other input can be safely cast to the input tensor dtype to prevent unexpected overflow or underflow (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, And(v["arg2_value"] > -128, v["arg2_value"] < 127), If(v["arg1_dtype"] == 2, And(v["arg2_value"] > -32768, v["arg2_value"] < 32767), If(v["arg1_dtype"] == 3, And(v["arg2_value"] > -2147483648, v["arg2_value"] < 2147483647), If(v["arg1_dtype"] == 4, And(v["arg2_value"] > -9223372036854775808, v["arg2_value"] < 9223372036854775807), If(v["arg1_dtype"] == 5, v["arg2_value"] > 0, False)))))) if n else
          If(v["arg1_dtype"] == 1, And(v["arg2_value"] > -128, v["arg2_value"] < 127), If(v["arg1_dtype"] == 2, And(v["arg2_value"] > -32768, v["arg2_value"] < 32767), If(v["arg1_dtype"] == 3, And(v["arg2_value"] > -2147483648, v["arg2_value"] < 2147483647), If(v["arg1_dtype"] == 4, And(v["arg2_value"] > -9223372036854775808, v["arg2_value"] < 9223372036854775807), If(v["arg1_dtype"] == 5, v["arg2_value"] > 0, False))))))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 87
        rule_87(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
