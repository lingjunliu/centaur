import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Target dimensions and offset dimensions can not be greater than the maximum allowed size for the given type of data. It can cause overflow errors. (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(v["arg5_value"] == 1, And(And(And(v["arg1_value"] < 128, v["arg2_value"] < 128), v["arg3_value"] < 128), v["arg4_value"] < 128), If(v["arg5_value"] == 2, And(And(And(v["arg1_value"] < 32768, v["arg2_value"] < 32768), v["arg3_value"] < 32768), v["arg4_value"] < 32768), If(v["arg5_value"] == 3, And(And(And(v["arg1_value"] < 2147483648, v["arg2_value"] < 2147483648), v["arg3_value"] < 2147483648), v["arg4_value"] < 2147483648), True)))) if n else
          If(v["arg5_value"] == 1, And(And(And(v["arg1_value"] < 128, v["arg2_value"] < 128), v["arg3_value"] < 128), v["arg4_value"] < 128), If(v["arg5_value"] == 2, And(And(And(v["arg1_value"] < 32768, v["arg2_value"] < 32768), v["arg3_value"] < 32768), v["arg4_value"] < 32768), If(v["arg5_value"] == 3, And(And(And(v["arg1_value"] < 2147483648, v["arg2_value"] < 2147483648), v["arg3_value"] < 2147483648), v["arg4_value"] < 2147483648), True))))
)

def rule_52_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, torch.dtype) or isinstance(arg5, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == list_of_available_dtypes.index(np_dtype(arg5)))

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
