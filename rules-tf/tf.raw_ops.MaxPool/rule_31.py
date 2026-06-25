import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If padding is EXPLICIT, the sum of explicit paddings for each dimension must be less than the input dimension size. (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == 33, v["arg1_length"] > 0), And(And(Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1) <= Select(v["arg3_shape"], 1), Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3) <= Select(v["arg3_shape"], 2)), Select(v["arg1_values"], 4) + Select(v["arg1_values"], 5) <= Select(v["arg3_shape"], 3)), True)) if n else
          If(And(v["arg2_value"] == 33, v["arg1_length"] > 0), And(And(Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1) <= Select(v["arg3_shape"], 1), Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3) <= Select(v["arg3_shape"], 2)), Select(v["arg1_values"], 4) + Select(v["arg1_values"], 5) <= Select(v["arg3_shape"], 3)), True))
)

def rule_31_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 31
        rule_31(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)
