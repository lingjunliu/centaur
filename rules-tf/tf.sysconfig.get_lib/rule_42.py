import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The returned library path depends on an integer, tensor, dtype, bool, float and string (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"], (If(v["arg2_dtype"] == v["arg3_value"], v["arg1_value"] * v["arg5_value"], v["arg1_value"] * v["arg5_value"])) >= v["arg1_value"] * v["arg5_value"], If(Or(Or(v["arg6_value"] == 0, v["arg6_value"] == 1), v["arg6_value"] == 2), (If(v["arg2_dtype"] == v["arg3_value"], v["arg1_value"] * v["arg5_value"], v["arg1_value"] * v["arg5_value"])) >= v["arg1_value"] * v["arg5_value"], (If(v["arg2_dtype"] == v["arg3_value"], v["arg1_value"] * v["arg5_value"], v["arg1_value"] * v["arg5_value"])) >= v["arg1_value"] * v["arg5_value"]))) if n else
          If(v["arg4_value"], (If(v["arg2_dtype"] == v["arg3_value"], v["arg1_value"] * v["arg5_value"], v["arg1_value"] * v["arg5_value"])) >= v["arg1_value"] * v["arg5_value"], If(Or(Or(v["arg6_value"] == 0, v["arg6_value"] == 1), v["arg6_value"] == 2), (If(v["arg2_dtype"] == v["arg3_value"], v["arg1_value"] * v["arg5_value"], v["arg1_value"] * v["arg5_value"])) >= v["arg1_value"] * v["arg5_value"], (If(v["arg2_dtype"] == v["arg3_value"], v["arg1_value"] * v["arg5_value"], v["arg1_value"] * v["arg5_value"])) >= v["arg1_value"] * v["arg5_value"])))
)

def rule_42_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False
        if not isinstance(arg6, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Real('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == list_of_string_values_tf.index(arg6))

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
