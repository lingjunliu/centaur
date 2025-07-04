import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If other is a dtype, then the input must be of a compatible data type family (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or((And(v["arg2_value"] == 0, v["arg1_dtype"] == 0)), (And(v["arg2_value"] == 12, v["arg1_dtype"] == 12))), (And(And(And(1 <= v["arg2_value"], v["arg2_value"] <= 5), 1 <= v["arg1_dtype"]), v["arg1_dtype"] <= 5))), (And(And(And(6 <= v["arg2_value"], v["arg2_value"] <= 8), 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 8))), (And(And(And(9 <= v["arg2_value"], v["arg2_value"] <= 10), 9 <= v["arg1_dtype"]), v["arg1_dtype"] <= 10)))) if n else
          Or(Or(Or(Or((And(v["arg2_value"] == 0, v["arg1_dtype"] == 0)), (And(v["arg2_value"] == 12, v["arg1_dtype"] == 12))), (And(And(And(1 <= v["arg2_value"], v["arg2_value"] <= 5), 1 <= v["arg1_dtype"]), v["arg1_dtype"] <= 5))), (And(And(And(6 <= v["arg2_value"], v["arg2_value"] <= 8), 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 8))), (And(And(And(9 <= v["arg2_value"], v["arg2_value"] <= 10), 9 <= v["arg1_dtype"]), v["arg1_dtype"] <= 10))))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 14
        rule_14(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
