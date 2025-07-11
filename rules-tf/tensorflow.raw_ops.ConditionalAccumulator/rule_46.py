import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# shape can be max rank 5, if dtype is complex then must be mean reduction type (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 9, v["arg2_value"] == 10), And(v["arg1_length"] <= 5, v["arg3_value"] == 7), v["arg1_length"] <= 5)) if n else
          If(Or(v["arg2_value"] == 9, v["arg2_value"] == 10), And(v["arg1_length"] <= 5, v["arg3_value"] == 7), v["arg1_length"] <= 5))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 46
        rule_46(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
