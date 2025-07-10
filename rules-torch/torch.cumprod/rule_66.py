import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the specified dtype is int32 or int64, the input dtype must be integer, floating point or complex, the input tensor must have values inside a valid range and the specified dtype is not boolean or string and the output tensor must have valid dtype (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 3, v["arg2_value"] == 4), And(And(And(And(And(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), Select(v["arg1_range"], 0) > -10000), Select(v["arg1_range"], 1) < 10000), v["arg2_value"] != 0), v["arg2_value"] != 11), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), v["arg3_dtype"] == 6), v["arg3_dtype"] == 7), v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), False)) if n else
          If(Or(v["arg2_value"] == 3, v["arg2_value"] == 4), And(And(And(And(And(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), Select(v["arg1_range"], 0) > -10000), Select(v["arg1_range"], 1) < 10000), v["arg2_value"] != 0), v["arg2_value"] != 11), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), v["arg3_dtype"] == 6), v["arg3_dtype"] == 7), v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), False))
)

def rule_66_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 66
        rule_66(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
