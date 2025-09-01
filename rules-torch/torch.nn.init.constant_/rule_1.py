import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Value should be within the valid range of the tensor's dtype to avoid overflow (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or((And(And(v["arg1_dtype"] == 7, v["arg2_value"] >= -65500), v["arg2_value"] <= 65500)), (And(And(v["arg1_dtype"] == 8, v["arg2_value"] >= -1e+38), v["arg2_value"] <= 1e+38))), (And(And(v["arg1_dtype"] == 1, v["arg2_value"] >= -128), v["arg2_value"] <= 127))), (And(And(v["arg1_dtype"] == 2, v["arg2_value"] >= -32768), v["arg2_value"] <= 32767))), (And(And(v["arg1_dtype"] == 3, v["arg2_value"] >= -2147483648), v["arg2_value"] <= 2147483647))), (And(And(v["arg1_dtype"] == 4, v["arg2_value"] >= -9223372036854775808), v["arg2_value"] <= 9223372036854775807))), (And(And(v["arg1_dtype"] == 6, v["arg2_value"] >= -65500), v["arg2_value"] <= 65500))), (And(And(v["arg1_dtype"] == 5, v["arg2_value"] >= 0), v["arg2_value"] <= 255)))) if n else
          Or(Or(Or(Or(Or(Or(Or((And(And(v["arg1_dtype"] == 7, v["arg2_value"] >= -65500), v["arg2_value"] <= 65500)), (And(And(v["arg1_dtype"] == 8, v["arg2_value"] >= -1e+38), v["arg2_value"] <= 1e+38))), (And(And(v["arg1_dtype"] == 1, v["arg2_value"] >= -128), v["arg2_value"] <= 127))), (And(And(v["arg1_dtype"] == 2, v["arg2_value"] >= -32768), v["arg2_value"] <= 32767))), (And(And(v["arg1_dtype"] == 3, v["arg2_value"] >= -2147483648), v["arg2_value"] <= 2147483647))), (And(And(v["arg1_dtype"] == 4, v["arg2_value"] >= -9223372036854775808), v["arg2_value"] <= 9223372036854775807))), (And(And(v["arg1_dtype"] == 6, v["arg2_value"] >= -65500), v["arg2_value"] <= 65500))), (And(And(v["arg1_dtype"] == 5, v["arg2_value"] >= 0), v["arg2_value"] <= 255))))
)

def rule_1_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (float, np.floating)) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 1
        rule_1(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
