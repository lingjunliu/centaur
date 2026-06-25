import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If ksize is [1,k,k,k,1] and strides is [1,s,s,s,1] and the padding is VALID then s should divide (input depth -k (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(And(And((And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 4) == 1)), (And(Select(v["arg2_values"], 0) == 1, Select(v["arg2_values"], 4) == 1))), (v["arg4_value"] == 28)), And(And((Select(v["arg3_shape"], 1) - Select(v["arg1_values"], 1)) % Select(v["arg2_values"], 1) == 0, (Select(v["arg3_shape"], 2) - Select(v["arg1_values"], 2)) % Select(v["arg2_values"], 2) == 0), (Select(v["arg3_shape"], 3) - Select(v["arg1_values"], 3)) % Select(v["arg2_values"], 3) == 0), True)) if n else
          If(And(And((And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 4) == 1)), (And(Select(v["arg2_values"], 0) == 1, Select(v["arg2_values"], 4) == 1))), (v["arg4_value"] == 28)), And(And((Select(v["arg3_shape"], 1) - Select(v["arg1_values"], 1)) % Select(v["arg2_values"], 1) == 0, (Select(v["arg3_shape"], 2) - Select(v["arg1_values"], 2)) % Select(v["arg2_values"], 2) == 0), (Select(v["arg3_shape"], 3) - Select(v["arg1_values"], 3)) % Select(v["arg2_values"], 3) == 0), True))
)

def rule_48_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 48
        rule_48(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)
