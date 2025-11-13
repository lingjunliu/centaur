import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if output dtype is provided then shape and rate must be of the same type (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), And(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), Or(Or(v["arg3_dtype"] == 6, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8)), If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), And(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4)), True))) if n else
          If(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), And(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), Or(Or(v["arg3_dtype"] == 6, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8)), If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), And(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4)), True)))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
