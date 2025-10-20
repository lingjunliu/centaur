import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If segment_ids are int64, then data tensor can be one of 'float32', 'float64', 'int32', 'uint8', 'int16', 'int8', 'int64', 'bfloat16', 'uint16', 'half', 'uint32', 'uint64' and if indices are int64, then data tensor can be one of 'float32', 'float64', 'int32', 'uint8', 'int16', 'int8', 'int64', 'bfloat16', 'uint16', 'half', 'uint32', 'uint64' (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 4, v["arg2_value"] == 4), Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 7, v["arg3_value"] == 8), v["arg3_value"] == 3), v["arg3_value"] == 5), v["arg3_value"] == 2), v["arg3_value"] == 1), v["arg3_value"] == 4), v["arg3_value"] == 6), v["arg3_value"] == 0), True)) if n else
          If(And(v["arg1_value"] == 4, v["arg2_value"] == 4), Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 7, v["arg3_value"] == 8), v["arg3_value"] == 3), v["arg3_value"] == 5), v["arg3_value"] == 2), v["arg3_value"] == 1), v["arg3_value"] == 4), v["arg3_value"] == 6), v["arg3_value"] == 0), True))
)

def rule_50_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
