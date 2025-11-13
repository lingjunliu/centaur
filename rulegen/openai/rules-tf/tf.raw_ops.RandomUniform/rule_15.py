import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# consistent RandomUniform configuration (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_ndim"] == 1, (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), (Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8))), Select(v["arg1_range"], 0) >= 0), -2147483648 <= v["arg3_value"]), v["arg3_value"] <= 2147483647), -2147483648 <= v["arg4_value"]), v["arg4_value"] <= 2147483647)) if n else
          And(And(And(And(And(And(And(v["arg1_ndim"] == 1, (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), (Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8))), Select(v["arg1_range"], 0) >= 0), -2147483648 <= v["arg3_value"]), v["arg3_value"] <= 2147483647), -2147483648 <= v["arg4_value"]), v["arg4_value"] <= 2147483647))
)

def rule_15_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 15
        rule_15(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
