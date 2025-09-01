import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If pad_width is -1, then max of ngram_widths should be less than the length of data (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == -1, (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) < Select(v["arg3_shape"], 0)) for i in range(6)])), True)) if n else
          If(v["arg1_value"] == -1, (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) < Select(v["arg3_shape"], 0)) for i in range(6)])), True))
)

def rule_18_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 18
        rule_18(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape']}, neg)
