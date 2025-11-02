import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If fast is true, A and B cannot be int8, int16, int32, int64, uint8 (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And((And(And(And(And(v["arg2_value"] != 1, v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5)), (And(And(And(And(v["arg3_value"] != 1, v["arg3_value"] != 2), v["arg3_value"] != 3), v["arg3_value"] != 4), v["arg3_value"] != 5))), True)) if n else
          If(v["arg1_value"] == True, And((And(And(And(And(v["arg2_value"] != 1, v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5)), (And(And(And(And(v["arg3_value"] != 1, v["arg3_value"] != 2), v["arg3_value"] != 3), v["arg3_value"] != 4), v["arg3_value"] != 5))), True))
)

def rule_31_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
