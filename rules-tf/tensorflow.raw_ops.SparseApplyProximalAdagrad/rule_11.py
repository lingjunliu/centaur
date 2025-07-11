import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, grad, lr, l1, l2 must be non-negative when their dtype is uint (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 18), v["arg1_dtype"] == 19), And(And(And(And(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg2_range"], 0) >= 0), Select(v["arg3_range"], 0) >= 0), Select(v["arg4_range"], 0) >= 0), Select(v["arg5_range"], 0) >= 0), Select(v["arg6_range"], 0) >= 0), False)) if n else
          If(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 18), v["arg1_dtype"] == 19), And(And(And(And(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg2_range"], 0) >= 0), Select(v["arg3_range"], 0) >= 0), Select(v["arg4_range"], 0) >= 0), Select(v["arg5_range"], 0) >= 0), Select(v["arg6_range"], 0) >= 0), False))
)

def rule_11_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
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
        if not isinstance(arg6, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_range = Array('arg5_range', IntSort(), IntSort())
        arg6_range = Array('arg6_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
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
        arg6_range = Store(arg6_range, 0, int(np.min(arg6)))
        arg6_range = Store(arg6_range, 1, int(np.max(arg6)))

        # Constraints for rule 11
        rule_11(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_range': arg4_range, 'arg5_range': arg5_range, 'arg6_range': arg6_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range'], 'arg5_range': arg5['range'], 'arg6_range': arg6['range']}, neg)
