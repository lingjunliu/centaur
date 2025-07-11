import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If reduction type is sum and dtype is float16 then shape.len should be 0 (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 8, v["arg2_value"] == 6), v["arg3_length"] == 0, False)) if n else
          If(And(v["arg1_value"] == 8, v["arg2_value"] == 6), v["arg3_length"] == 0, False))
)

def rule_55_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 55
        rule_55(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
