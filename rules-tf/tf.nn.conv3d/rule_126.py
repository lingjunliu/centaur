import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# first and last elements of strides must be 1 depending on data format (Rule 126)

rule_126 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg3_ndim"] == 5, v["arg1_length"] == 5), (If(v["arg2_value"] == 25, And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 1) == 1), And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 4) == 1))))) if n else
          And(And(v["arg3_ndim"] == 5, v["arg1_length"] == 5), (If(v["arg2_value"] == 25, And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 1) == 1), And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 4) == 1)))))
)

def rule_126_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 126
        rule_126(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
