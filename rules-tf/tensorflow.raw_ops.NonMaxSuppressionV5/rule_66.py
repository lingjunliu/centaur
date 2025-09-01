import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Comprehensive validation with score_threshold lower bound (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) > 0, v["arg2_ndim"] == 1), v["arg1_ndim"] == 2), Select(v["arg1_shape"], 1) == 4), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), Select(v["arg3_range"], 0) > 0), Select(v["arg4_range"], 0) >= 0.0), Select(v["arg4_range"], 1) <= 1.0), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6)))), Select(v["arg6_range"], 0) >= 0.0), (If(v["arg7_value"] == True, Select(v["arg3_range"], 0) > 0, True))), (If(Select(v["arg2_shape"], 0) > 0, Select(v["arg5_range"], 0) >= (Select(v["arg2_range"], 0) / 2), True)))) if n else
          And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) > 0, v["arg2_ndim"] == 1), v["arg1_ndim"] == 2), Select(v["arg1_shape"], 1) == 4), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), Select(v["arg3_range"], 0) > 0), Select(v["arg4_range"], 0) >= 0.0), Select(v["arg4_range"], 1) <= 1.0), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6)))), Select(v["arg6_range"], 0) >= 0.0), (If(v["arg7_value"] == True, Select(v["arg3_range"], 0) > 0, True))), (If(Select(v["arg2_shape"], 0) > 0, Select(v["arg5_range"], 0) >= (Select(v["arg2_range"], 0) / 2), True))))
)

def rule_66_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

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
        if not isinstance(arg7, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_range = Array('arg5_range', IntSort(), IntSort())
        arg6_range = Array('arg6_range', IntSort(), IntSort())
        arg7_value = Bool('arg7_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
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
        solver.add(arg7_value == arg7)

        # Constraints for rule 66
        rule_66(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_range': arg4_range, 'arg5_range': arg5_range, 'arg6_range': arg6_range, 'arg7_value': arg7_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range'], 'arg5_range': arg5['range'], 'arg6_range': arg6['range'], 'arg7_value': arg7['value']}, neg)
