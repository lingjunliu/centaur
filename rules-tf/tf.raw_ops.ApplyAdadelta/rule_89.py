import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If use_locking is True then all tensors dimensions should be greater than 0, otherwise, it does not matter (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, (And(And(And(And(And(And(Select(v["arg2_range"], 0) > 0, Select(v["arg3_range"], 0) > 0), Select(v["arg4_range"], 0) > 0), Select(v["arg5_range"], 0) > 0), Select(v["arg6_range"], 0) > 0), Select(v["arg7_range"], 0) > 0), Select(v["arg8_range"], 0) > 0)), True)) if n else
          If(v["arg1_value"] == True, (And(And(And(And(And(And(Select(v["arg2_range"], 0) > 0, Select(v["arg3_range"], 0) > 0), Select(v["arg4_range"], 0) > 0), Select(v["arg5_range"], 0) > 0), Select(v["arg6_range"], 0) > 0), Select(v["arg7_range"], 0) > 0), Select(v["arg8_range"], 0) > 0)), True))
)

def rule_89_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, np.ndarray):
            return False
        if not isinstance(arg7, np.ndarray):
            return False
        if not isinstance(arg8, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_range = Array('arg5_range', IntSort(), IntSort())
        arg6_range = Array('arg6_range', IntSort(), IntSort())
        arg7_range = Array('arg7_range', IntSort(), IntSort())
        arg8_range = Array('arg8_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))
        arg5_range = Store(arg5_range, 0, int(np.min(arg5)))
        arg5_range = Store(arg5_range, 1, int(np.max(arg5)))
        arg6_range = Store(arg6_range, 0, int(np.min(arg6)))
        arg6_range = Store(arg6_range, 1, int(np.max(arg6)))
        arg7_range = Store(arg7_range, 0, int(np.min(arg7)))
        arg7_range = Store(arg7_range, 1, int(np.max(arg7)))
        arg8_range = Store(arg8_range, 0, int(np.min(arg8)))
        arg8_range = Store(arg8_range, 1, int(np.max(arg8)))

        # Constraints for rule 89
        rule_89(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_range': arg4_range, 'arg5_range': arg5_range, 'arg6_range': arg6_range, 'arg7_range': arg7_range, 'arg8_range': arg8_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range'], 'arg5_range': arg5['range'], 'arg6_range': arg6['range'], 'arg7_range': arg7['range'], 'arg8_range': arg8['range']}, neg)
