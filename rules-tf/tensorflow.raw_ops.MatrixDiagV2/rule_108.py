import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If k is a tuple, its values must be less than num_rows or num_cols if they are given and padding value must be integer (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(And(And((And(Select(v["arg1_values"], 0) < v["arg2_value"], Or(Select(v["arg1_values"], 1) < v["arg2_value"], v["arg2_value"] == 0))), (And(Select(v["arg1_values"], 0) < v["arg3_value"], Or(Select(v["arg1_values"], 1) < v["arg3_value"], v["arg3_value"] == 0)))), (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5)))) if n else
          And(And((And(Select(v["arg1_values"], 0) < v["arg2_value"], Or(Select(v["arg1_values"], 1) < v["arg2_value"], v["arg2_value"] == 0))), (And(Select(v["arg1_values"], 0) < v["arg3_value"], Or(Select(v["arg1_values"], 1) < v["arg3_value"], v["arg3_value"] == 0)))), (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5))))
)

def rule_108_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 108
        rule_108(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
