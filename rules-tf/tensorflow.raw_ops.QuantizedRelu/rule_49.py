import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# features dtype and out_type dtype must be a quantized type and min_features and max_features are float32 (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 5), v["arg4_value"] == 3), v["arg4_value"] == 2), v["arg4_value"] == 14)), (Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 3), v["arg1_dtype"] == 2), v["arg1_dtype"] == 14))), v["arg2_dtype"] == 7), v["arg3_dtype"] == 7)) if n else
          And(And(And((Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 5), v["arg4_value"] == 3), v["arg4_value"] == 2), v["arg4_value"] == 14)), (Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 3), v["arg1_dtype"] == 2), v["arg1_dtype"] == 14))), v["arg2_dtype"] == 7), v["arg3_dtype"] == 7))
)

def rule_49_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 49
        rule_49(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
