import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype of the first tensor is equal to 7, the max value in the tuple needs to be less than the max value of the first tensor. (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 7, (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) < Select(v["arg1_range"], 1)) for i in range(6)])), True)) if n else
          If(v["arg3_value"] == 7, (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) < Select(v["arg1_range"], 1)) for i in range(6)])), True))
)

def rule_81_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 81
        rule_81(solver, {'arg1_range': arg1_range, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_range': arg1['range'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
