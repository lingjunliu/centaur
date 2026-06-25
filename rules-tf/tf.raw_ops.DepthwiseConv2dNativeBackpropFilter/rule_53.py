import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input and out_backprop tensors have float64 type, explicit padding must be zero (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8), v["arg4_value"] == 30), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) == 0) for i in range(6)]), True)) if n else
          If(And(And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8), v["arg4_value"] == 30), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) == 0) for i in range(6)]), True))
)

def rule_53_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 53
        rule_53(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value']}, neg)
