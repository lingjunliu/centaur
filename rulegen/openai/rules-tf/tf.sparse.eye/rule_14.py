import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tensor num_columns: scalar, non-negative; dtype must be a valid TF dtype; bounded size (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_value"] >= 0, v["arg2_ndim"] == 0), Select(v["arg2_range"], 0) >= 0), v["arg3_value"] == v["arg2_dtype"]), v["arg1_value"] * Select(v["arg2_range"], 0) <= 100000000), v["arg4_value"] != 6)) if n else
          And(And(And(And(And(v["arg1_value"] >= 0, v["arg2_ndim"] == 0), Select(v["arg2_range"], 0) >= 0), v["arg3_value"] == v["arg2_dtype"]), v["arg1_value"] * Select(v["arg2_range"], 0) <= 100000000), v["arg4_value"] != 6))
)

def rule_14_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
