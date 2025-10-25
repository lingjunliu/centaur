import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All parameters should have valid values and dtype should be float type (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg3_value"] > 0), v["arg4_value"] >= 0), v["arg5_value"] > v["arg4_value"]), (Or(Or(v["arg6_value"] == 6, v["arg6_value"] == 7), v["arg6_value"] == 8)))) if n else
          And(And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg3_value"] > 0), v["arg4_value"] >= 0), v["arg5_value"] > v["arg4_value"]), (Or(Or(v["arg6_value"] == 6, v["arg6_value"] == 7), v["arg6_value"] == 8))))
)

def rule_75_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not ((isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)) or isinstance(arg4, (float, np.floating))):
            return False
        if not ((isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)) or isinstance(arg5, (float, np.floating))):
            return False
        if not (isinstance(arg6, torch.dtype) or isinstance(arg6, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg3_value == int(arg3))
        solver.add(arg6_value == list_of_available_dtypes.index(np_dtype(arg6)))

        # Constraints for rule 75
        rule_75(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
