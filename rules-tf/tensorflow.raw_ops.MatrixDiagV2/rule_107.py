import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If k is a scalar, it must be less than num_rows or num_cols if they are given but at least must be an int32 representable and padding value must be integer (Rule 107)

rule_107 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(v["arg1_value"] < v["arg2_value"], v["arg2_value"] == 0)), (Or(v["arg1_value"] < v["arg3_value"], v["arg3_value"] == 0))), v["arg1_value"] >= -2147483648), v["arg1_value"] <= 2147483647), (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5)))) if n else
          And(And(And(And((Or(v["arg1_value"] < v["arg2_value"], v["arg2_value"] == 0)), (Or(v["arg1_value"] < v["arg3_value"], v["arg3_value"] == 0))), v["arg1_value"] >= -2147483648), v["arg1_value"] <= 2147483647), (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5))))
)

def rule_107_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 107
        rule_107(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
