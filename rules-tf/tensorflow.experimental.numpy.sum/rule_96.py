import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The dtypes for the tensor 'a' and the provided 'dtype' must be compatible (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 1, (Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), If(v["arg2_value"] == 2, (Or(Or(Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), If(v["arg2_value"] == 3, (Or(Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), If(v["arg2_value"] == 4, (Or(v["arg1_dtype"] == 4, v["arg1_dtype"] == 5)), If(v["arg2_value"] == 5, v["arg1_dtype"] == 5, If(v["arg2_value"] == 6, (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), If(v["arg2_value"] == 7, (Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), If(v["arg2_value"] == 8, v["arg1_dtype"] == 8, If(v["arg2_value"] == 9, (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), If(v["arg2_value"] == 10, v["arg1_dtype"] == 10, If(v["arg2_value"] == 0, True, If(v["arg2_value"] == 12, True, True))))))))))))) if n else
          If(v["arg2_value"] == 1, (Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), If(v["arg2_value"] == 2, (Or(Or(Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), If(v["arg2_value"] == 3, (Or(Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), If(v["arg2_value"] == 4, (Or(v["arg1_dtype"] == 4, v["arg1_dtype"] == 5)), If(v["arg2_value"] == 5, v["arg1_dtype"] == 5, If(v["arg2_value"] == 6, (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), If(v["arg2_value"] == 7, (Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), If(v["arg2_value"] == 8, v["arg1_dtype"] == 8, If(v["arg2_value"] == 9, (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), If(v["arg2_value"] == 10, v["arg1_dtype"] == 10, If(v["arg2_value"] == 0, True, If(v["arg2_value"] == 12, True, True)))))))))))))
)

def rule_96_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 96
        rule_96(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
