import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape.len == 0, then dtype cannot be type bool, int8, int16, int32, int64, uint8, qint8, quint8, qint16, quint16, qint32, or str (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 0, And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 0, v["arg2_value"] != 1), v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5), v["arg2_value"] != 12), v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 13), v["arg2_value"] != 14), False)) if n else
          If(v["arg1_length"] == 0, And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 0, v["arg2_value"] != 1), v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5), v["arg2_value"] != 12), v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 13), v["arg2_value"] != 14), False))
)

def rule_72_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 72
        rule_72(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
