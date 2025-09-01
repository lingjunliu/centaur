import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If features is qint32, and out_type is quint16, then min_features must be greater or equal to zero, and max_features should be less or equal to 65535. (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 3, v["arg4_value"] == 4), And(Select(v["arg2_range"], 0) >= 0, Select(v["arg3_range"], 1) <= 65535), True)) if n else
          If(And(v["arg1_dtype"] == 3, v["arg4_value"] == 4), And(Select(v["arg2_range"], 0) >= 0, Select(v["arg3_range"], 1) <= 65535), True))
)

def rule_60_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 60
        rule_60(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_value': arg4['value']}, neg)
