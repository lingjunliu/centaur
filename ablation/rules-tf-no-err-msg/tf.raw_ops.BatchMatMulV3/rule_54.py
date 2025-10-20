import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x's dtype is float32/float64/complex64/complex128 and y's dtype is uint8/int8/int16/int32/int64, then tout should be float32/float64/complex64/complex128 (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Or(Or(Or(Or(v["arg2_dtype"] == 2, v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), v["arg2_dtype"] == 6))), (Or(Or(Or(Or(v["arg3_value"] == 7, v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 12)), True)) if n else
          If(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Or(Or(Or(Or(v["arg2_dtype"] == 2, v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), v["arg2_dtype"] == 6))), (Or(Or(Or(Or(v["arg3_value"] == 7, v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 12)), True))
)

def rule_54_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 54
        rule_54(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
