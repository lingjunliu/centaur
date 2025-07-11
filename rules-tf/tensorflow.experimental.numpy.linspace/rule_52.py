import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The dtype should be compatible with start and stop if they are tensors. (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If((Or(Or(Or(Or(v["arg3_value"] == 1, v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5)), And((Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), (Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5))), If((Or(Or(v["arg3_value"] == 6, v["arg3_value"] == 7), v["arg3_value"] == 8)), And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), (Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8))), False))) if n else
          If((Or(Or(Or(Or(v["arg3_value"] == 1, v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5)), And((Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), (Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5))), If((Or(Or(v["arg3_value"] == 6, v["arg3_value"] == 7), v["arg3_value"] == 8)), And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), (Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8))), False)))
)

def rule_52_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 52
        rule_52(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
