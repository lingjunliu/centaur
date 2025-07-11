import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If binary_output is True, then the output dtype is int32, and weights is None and arr is not empty and arr is integer type (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, (And(1 <= v["arg3_dtype"], v["arg3_dtype"] <= 5))), v["arg2_value"] == 3, False)) if n else
          If(And(v["arg1_value"] == True, (And(1 <= v["arg3_dtype"], v["arg3_dtype"] <= 5))), v["arg2_value"] == 3, False))
)

def rule_64_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 64
        rule_64(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
