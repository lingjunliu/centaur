import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if start, end, or step are float, then dtype must be float (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or((v["arg4_value"] == 7), (v["arg4_value"] == 8)), (v["arg4_value"] == 6)), True, (If(Or(Or((v["arg1_value"] > 0.0), (v["arg2_value"] > 0.0)), (v["arg3_value"] > 0.0)), Or(Or((v["arg4_value"] == 7), (v["arg4_value"] == 8)), (v["arg4_value"] == 6)), False)))) if n else
          If(Or(Or((v["arg4_value"] == 7), (v["arg4_value"] == 8)), (v["arg4_value"] == 6)), True, (If(Or(Or((v["arg1_value"] > 0.0), (v["arg2_value"] > 0.0)), (v["arg3_value"] > 0.0)), Or(Or((v["arg4_value"] == 7), (v["arg4_value"] == 8)), (v["arg4_value"] == 6)), False))))
)

def rule_9_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

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

        # Variable declarations
        solver = Solver()
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
