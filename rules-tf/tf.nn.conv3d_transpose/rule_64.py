import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data_format is NCDHW, then dilations dim should match input dim size except Batch and Channel, and dim values at 0 and 1 position must be 1 (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] == 5, v["arg3_value"] == 36), And(And(And(And(Select(v["arg2_shape"], 2) > 0, Select(v["arg2_shape"], 3) > 0), Select(v["arg2_shape"], 4) > 0), Select(v["arg1_values"], 0) == 1), Select(v["arg1_values"], 1) == 1), True)) if n else
          If(And(v["arg1_length"] == 5, v["arg3_value"] == 36), And(And(And(And(Select(v["arg2_shape"], 2) > 0, Select(v["arg2_shape"], 3) > 0), Select(v["arg2_shape"], 4) > 0), Select(v["arg1_values"], 0) == 1), Select(v["arg1_values"], 1) == 1), True))
)

def rule_64_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 64
        rule_64(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
