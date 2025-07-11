import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All floating-point tensors (var, accum, linear, grad, lr, l1, l2, lr_power (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), v["arg1_dtype"] == v["arg5_dtype"]), v["arg1_dtype"] == v["arg6_dtype"]), v["arg1_dtype"] == v["arg7_dtype"]), v["arg1_dtype"] == v["arg8_dtype"]), v["arg1_ndim"] > 0), v["arg2_ndim"] > 0), v["arg3_ndim"] > 0), v["arg4_ndim"] > 0), v["arg5_ndim"] == 0), v["arg6_ndim"] == 0), v["arg7_ndim"] == 0), v["arg8_ndim"] == 0), Select(v["arg5_range"], 0) >= 0), Select(v["arg6_range"], 0) >= 0), Select(v["arg7_range"], 0) >= 0), Select(v["arg8_range"], 0) >= 0), Select(v["arg8_range"], 0) >= 0), Select(v["arg8_range"], 1) <= 1)) if n else
          And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), v["arg1_dtype"] == v["arg5_dtype"]), v["arg1_dtype"] == v["arg6_dtype"]), v["arg1_dtype"] == v["arg7_dtype"]), v["arg1_dtype"] == v["arg8_dtype"]), v["arg1_ndim"] > 0), v["arg2_ndim"] > 0), v["arg3_ndim"] > 0), v["arg4_ndim"] > 0), v["arg5_ndim"] == 0), v["arg6_ndim"] == 0), v["arg7_ndim"] == 0), v["arg8_ndim"] == 0), Select(v["arg5_range"], 0) >= 0), Select(v["arg6_range"], 0) >= 0), Select(v["arg7_range"], 0) >= 0), Select(v["arg8_range"], 0) >= 0), Select(v["arg8_range"], 0) >= 0), Select(v["arg8_range"], 1) <= 1))
)

def rule_75_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
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
        if not isinstance(arg7, np.ndarray):
            return False
        if not isinstance(arg8, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_dtype = Int('arg4_dtype')
        arg5_ndim = Int('arg5_ndim')
        arg5_dtype = Int('arg5_dtype')
        arg5_range = Array('arg5_range', IntSort(), IntSort())
        arg6_ndim = Int('arg6_ndim')
        arg6_dtype = Int('arg6_dtype')
        arg6_range = Array('arg6_range', IntSort(), IntSort())
        arg7_ndim = Int('arg7_ndim')
        arg7_dtype = Int('arg7_dtype')
        arg7_range = Array('arg7_range', IntSort(), IntSort())
        arg8_ndim = Int('arg8_ndim')
        arg8_dtype = Int('arg8_dtype')
        arg8_range = Array('arg8_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))
        arg5_range = Store(arg5_range, 0, int(np.min(arg5)))
        arg5_range = Store(arg5_range, 1, int(np.max(arg5)))
        solver.add(arg6_ndim == arg6.ndim)
        solver.add(arg6_dtype == list_of_available_dtypes.index(arg6.dtype))
        arg6_range = Store(arg6_range, 0, int(np.min(arg6)))
        arg6_range = Store(arg6_range, 1, int(np.max(arg6)))
        solver.add(arg7_ndim == arg7.ndim)
        solver.add(arg7_dtype == list_of_available_dtypes.index(arg7.dtype))
        arg7_range = Store(arg7_range, 0, int(np.min(arg7)))
        arg7_range = Store(arg7_range, 1, int(np.max(arg7)))
        solver.add(arg8_ndim == arg8.ndim)
        solver.add(arg8_dtype == list_of_available_dtypes.index(arg8.dtype))
        arg8_range = Store(arg8_range, 0, int(np.min(arg8)))
        arg8_range = Store(arg8_range, 1, int(np.max(arg8)))

        # Constraints for rule 75
        rule_75(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg4_dtype': arg4_dtype, 'arg4_ndim': arg4_ndim, 'arg5_dtype': arg5_dtype, 'arg5_range': arg5_range, 'arg5_ndim': arg5_ndim, 'arg6_dtype': arg6_dtype, 'arg6_range': arg6_range, 'arg6_ndim': arg6_ndim, 'arg7_dtype': arg7_dtype, 'arg7_range': arg7_range, 'arg7_ndim': arg7_ndim, 'arg8_dtype': arg8_dtype, 'arg8_range': arg8_range, 'arg8_ndim': arg8_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg4_dtype': arg4['dtype'], 'arg4_ndim': arg4['ndim'], 'arg5_dtype': arg5['dtype'], 'arg5_range': arg5['range'], 'arg5_ndim': arg5['ndim'], 'arg6_dtype': arg6['dtype'], 'arg6_range': arg6['range'], 'arg6_ndim': arg6['ndim'], 'arg7_dtype': arg7['dtype'], 'arg7_range': arg7['range'], 'arg7_ndim': arg7['ndim'], 'arg8_dtype': arg8['dtype'], 'arg8_range': arg8['range'], 'arg8_ndim': arg8['ndim']}, neg)
