import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Dtype is correctly inferred (Rule 110)

rule_110 = lambda s, v, n=False: (
    s.add(Not(If(v["arg5_value"], (If(Or(Or((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), (Or(v["arg4_value"] == 7, v["arg4_value"] == 8)), (Or(v["arg4_value"] == 9, v["arg4_value"] == 10)))), If(Or(Or((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), (Or(v["arg4_value"] == 7, v["arg4_value"] == 8)), (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5))))) if n else
          If(v["arg5_value"], (If(Or(Or((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), (Or(v["arg4_value"] == 7, v["arg4_value"] == 8)), (Or(v["arg4_value"] == 9, v["arg4_value"] == 10)))), If(Or(Or((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), (Or(v["arg3_value"] == 7, v["arg3_value"] == 8))), (Or(v["arg4_value"] == 7, v["arg4_value"] == 8)), (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5)))))
)

def rule_110_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg4_value = Int('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))
        solver.add(arg5_value == arg5)

        # Constraints for rule 110
        rule_110(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_110(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
