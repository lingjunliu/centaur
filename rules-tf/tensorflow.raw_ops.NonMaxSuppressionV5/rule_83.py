import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All thresholds and box data type (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg2_range"], 0) >= 0.0, Select(v["arg2_range"], 1) <= 1.0), Select(v["arg3_range"], 0) >= 0), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6)))), Select(v["arg4_range"], 0) >= 0.0), (If(v["arg5_value"] == True, Select(v["arg6_range"], 0) > 0, True)))) if n else
          And(And(And(And(And(Select(v["arg2_range"], 0) >= 0.0, Select(v["arg2_range"], 1) <= 1.0), Select(v["arg3_range"], 0) >= 0), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6)))), Select(v["arg4_range"], 0) >= 0.0), (If(v["arg5_value"] == True, Select(v["arg6_range"], 0) > 0, True))))
)

def rule_83_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
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
        if not isinstance(arg5, bool):
            return False
        if not isinstance(arg6, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_value = Bool('arg5_value')
        arg6_range = Array('arg6_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))
        solver.add(arg5_value == arg5)
        arg6_range = Store(arg6_range, 0, int(np.min(arg6)))
        arg6_range = Store(arg6_range, 1, int(np.max(arg6)))

        # Constraints for rule 83
        rule_83(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_range': arg4_range, 'arg5_value': arg5_value, 'arg6_range': arg6_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range'], 'arg5_value': arg5['value'], 'arg6_range': arg6['range']}, neg)
