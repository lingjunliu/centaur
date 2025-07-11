import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the output dtype is provided as qint32, x and y should be qint32 and the min of x should be less than or equal to zero (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 3, And(And(v["arg2_dtype"] == 3, v["arg3_dtype"] == 3), Select(v["arg2_range"], 0) <= 0), False)) if n else
          If(v["arg1_value"] == 3, And(And(v["arg2_dtype"] == 3, v["arg3_dtype"] == 3), Select(v["arg2_range"], 0) <= 0), False))
)

def rule_65_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg3_dtype': arg3['dtype']}, neg)
