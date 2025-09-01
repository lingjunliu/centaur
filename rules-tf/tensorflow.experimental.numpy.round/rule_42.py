import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# decimals must be an integer between -30 and 30, and if the input is boolean, then decimals must be 0, if the input is an integer tensor, decimals must be non-negative, and decimals should not be greater than the maximum representable precision of the floating point tensor (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg2_value"] >= -30, v["arg2_value"] <= 30), (If(v["arg1_dtype"] == 0, v["arg2_value"] == 0, True))), (If(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg2_value"] >= 0, True))), (If(v["arg1_dtype"] == 6, v["arg2_value"] <= 3, If(v["arg1_dtype"] == 7, v["arg2_value"] <= 7, If(v["arg1_dtype"] == 8, v["arg2_value"] <= 15, If(v["arg1_dtype"] == 9, v["arg2_value"] <= 7, If(v["arg1_dtype"] == 10, v["arg2_value"] <= 15, True)))))))) if n else
          And(And(And(And(v["arg2_value"] >= -30, v["arg2_value"] <= 30), (If(v["arg1_dtype"] == 0, v["arg2_value"] == 0, True))), (If(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg2_value"] >= 0, True))), (If(v["arg1_dtype"] == 6, v["arg2_value"] <= 3, If(v["arg1_dtype"] == 7, v["arg2_value"] <= 7, If(v["arg1_dtype"] == 8, v["arg2_value"] <= 15, If(v["arg1_dtype"] == 9, v["arg2_value"] <= 7, If(v["arg1_dtype"] == 10, v["arg2_value"] <= 15, True))))))))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 42
        rule_42(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
