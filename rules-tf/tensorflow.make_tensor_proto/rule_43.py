import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The dtype and values should be compatible. (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, v["arg2_value"] == 0, If(v["arg1_dtype"] == 1, v["arg2_value"] == 1, If(v["arg1_dtype"] == 2, v["arg2_value"] == 2, If(v["arg1_dtype"] == 3, v["arg2_value"] == 3, If(v["arg1_dtype"] == 4, v["arg2_value"] == 4, If(v["arg1_dtype"] == 5, v["arg2_value"] == 5, If(v["arg1_dtype"] == 6, v["arg2_value"] == 6, If(v["arg1_dtype"] == 7, v["arg2_value"] == 7, If(v["arg1_dtype"] == 8, v["arg2_value"] == 8, If(v["arg1_dtype"] == 9, v["arg2_value"] == 9, If(v["arg1_dtype"] == 10, v["arg2_value"] == 10, If(v["arg1_dtype"] == 11, v["arg2_value"] == 11, If(v["arg1_dtype"] == 12, v["arg2_value"] == 12, True)))))))))))))) if n else
          If(v["arg1_dtype"] == 0, v["arg2_value"] == 0, If(v["arg1_dtype"] == 1, v["arg2_value"] == 1, If(v["arg1_dtype"] == 2, v["arg2_value"] == 2, If(v["arg1_dtype"] == 3, v["arg2_value"] == 3, If(v["arg1_dtype"] == 4, v["arg2_value"] == 4, If(v["arg1_dtype"] == 5, v["arg2_value"] == 5, If(v["arg1_dtype"] == 6, v["arg2_value"] == 6, If(v["arg1_dtype"] == 7, v["arg2_value"] == 7, If(v["arg1_dtype"] == 8, v["arg2_value"] == 8, If(v["arg1_dtype"] == 9, v["arg2_value"] == 9, If(v["arg1_dtype"] == 10, v["arg2_value"] == 10, If(v["arg1_dtype"] == 11, v["arg2_value"] == 11, If(v["arg1_dtype"] == 12, v["arg2_value"] == 12, True))))))))))))))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 43
        rule_43(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
