import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is specified, the start, end, step params should be compatible, and out tensor should have the same specified dtype (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(v["arg5_value"] != 0, (And((If(v["arg5_value"] <= 5, (And(And(v["arg1_value"] % 1 == 0, v["arg2_value"] % 1 == 0), v["arg3_value"] % 1 == 0)), True)), v["arg4_dtype"] == v["arg5_value"])), True)) if n else
          If(v["arg5_value"] != 0, (And((If(v["arg5_value"] <= 5, (And(And(v["arg1_value"] % 1 == 0, v["arg2_value"] % 1 == 0), v["arg3_value"] % 1 == 0)), True)), v["arg4_dtype"] == v["arg5_value"])), True))
)

def rule_31_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not (isinstance(arg5, torch.dtype) or isinstance(arg5, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_dtype = Int('arg4_dtype')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_value == list_of_available_dtypes.index(np_dtype(arg5)))

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype'], 'arg5_value': arg5['value']}, neg)
