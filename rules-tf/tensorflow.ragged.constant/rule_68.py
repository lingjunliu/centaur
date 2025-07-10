import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Check that the values in pylist match the provided type index (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And((If(v["arg1_value"] == 0, v["arg2_dtype"] == 0, False)), (If(v["arg1_value"] == 1, v["arg2_dtype"] == 1, False))), (If(v["arg1_value"] == 2, v["arg2_dtype"] == 2, False))), (If(v["arg1_value"] == 3, v["arg2_dtype"] == 3, False))), (If(v["arg1_value"] == 4, v["arg2_dtype"] == 4, False))), (If(v["arg1_value"] == 5, v["arg2_dtype"] == 5, False))), (If(v["arg1_value"] == 6, v["arg2_dtype"] == 6, False))), (If(v["arg1_value"] == 7, v["arg2_dtype"] == 7, False))), (If(v["arg1_value"] == 8, v["arg2_dtype"] == 8, False))), (If(v["arg1_value"] == 9, v["arg2_dtype"] == 9, False))), (If(v["arg1_value"] == 10, v["arg2_dtype"] == 10, False))), (If(v["arg1_value"] == 11, v["arg2_dtype"] == 11, False))), (If(v["arg1_value"] == 12, v["arg2_dtype"] == 12, False)))) if n else
          And(And(And(And(And(And(And(And(And(And(And(And((If(v["arg1_value"] == 0, v["arg2_dtype"] == 0, False)), (If(v["arg1_value"] == 1, v["arg2_dtype"] == 1, False))), (If(v["arg1_value"] == 2, v["arg2_dtype"] == 2, False))), (If(v["arg1_value"] == 3, v["arg2_dtype"] == 3, False))), (If(v["arg1_value"] == 4, v["arg2_dtype"] == 4, False))), (If(v["arg1_value"] == 5, v["arg2_dtype"] == 5, False))), (If(v["arg1_value"] == 6, v["arg2_dtype"] == 6, False))), (If(v["arg1_value"] == 7, v["arg2_dtype"] == 7, False))), (If(v["arg1_value"] == 8, v["arg2_dtype"] == 8, False))), (If(v["arg1_value"] == 9, v["arg2_dtype"] == 9, False))), (If(v["arg1_value"] == 10, v["arg2_dtype"] == 10, False))), (If(v["arg1_value"] == 11, v["arg2_dtype"] == 11, False))), (If(v["arg1_value"] == 12, v["arg2_dtype"] == 12, False))))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
