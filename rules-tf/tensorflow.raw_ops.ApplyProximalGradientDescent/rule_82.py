import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If locking is used, alpha, l1, l2, delta and var shouldn't have extreme values. (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(If(v["arg6_value"] == True, And(And(And(And(Select(v["arg1_range"], 1) < 1000000, Select(v["arg2_range"], 1) < 1000000), Select(v["arg3_range"], 1) < 1000000), Select(v["arg4_range"], 1) < 1000000), Select(v["arg5_range"], 1) < 1000000), False)) if n else
          If(v["arg6_value"] == True, And(And(And(And(Select(v["arg1_range"], 1) < 1000000, Select(v["arg2_range"], 1) < 1000000), Select(v["arg3_range"], 1) < 1000000), Select(v["arg4_range"], 1) < 1000000), Select(v["arg5_range"], 1) < 1000000), False))
)

def rule_82_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_range = Array('arg5_range', IntSort(), IntSort())
        arg6_value = Bool('arg6_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))
        arg5_range = Store(arg5_range, 0, int(np.min(arg5)))
        arg5_range = Store(arg5_range, 1, int(np.max(arg5)))
        solver.add(arg6_value == arg6)

        # Constraints for rule 82
        rule_82(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_range': arg4_range, 'arg5_range': arg5_range, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range'], 'arg5_range': arg5['range'], 'arg6_value': arg6['value']}, neg)
