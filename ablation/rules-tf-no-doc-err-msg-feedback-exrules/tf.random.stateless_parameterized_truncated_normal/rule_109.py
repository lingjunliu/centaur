import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a shape is provided then mean, stddev, truncated_threshold should have dtypes that are float or the specified dtype (Rule 109)

rule_109 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg5_ndim"] > 0, v["arg1_value"] != 0)), And(And((Or(Or(Or((v["arg2_dtype"] == v["arg1_value"]), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 6))), (Or(Or(Or((v["arg3_dtype"] == v["arg1_value"]), (v["arg3_dtype"] == 7)), (v["arg3_dtype"] == 8)), (v["arg3_dtype"] == 6)))), (Or(Or(Or((v["arg4_dtype"] == v["arg1_value"]), (v["arg4_dtype"] == 7)), (v["arg4_dtype"] == 8)), (v["arg4_dtype"] == 6)))), True)) if n else
          If((And(v["arg5_ndim"] > 0, v["arg1_value"] != 0)), And(And((Or(Or(Or((v["arg2_dtype"] == v["arg1_value"]), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 6))), (Or(Or(Or((v["arg3_dtype"] == v["arg1_value"]), (v["arg3_dtype"] == 7)), (v["arg3_dtype"] == 8)), (v["arg3_dtype"] == 6)))), (Or(Or(Or((v["arg4_dtype"] == v["arg1_value"]), (v["arg4_dtype"] == 7)), (v["arg4_dtype"] == 8)), (v["arg4_dtype"] == 6)))), True))
)

def rule_109_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 109
        rule_109(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_ndim': arg5['ndim']}, neg)
