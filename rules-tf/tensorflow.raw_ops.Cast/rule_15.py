import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If truncate is true, and casting from float to int, destination must be int (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == True, (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6))), And(v["arg2_value"] <= 5, v["arg2_value"] >= 1), False)) if n else
          If(And(v["arg3_value"] == True, (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6))), And(v["arg2_value"] <= 5, v["arg2_value"] >= 1), False))
)

def rule_15_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 15
        rule_15(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
