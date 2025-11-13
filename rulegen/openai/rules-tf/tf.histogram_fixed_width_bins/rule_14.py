import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# list value_range with floats: values must be floating-point and interval ordered (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And(And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg2_length"] == 2), Select(v["arg2_values"], 0) < Select(v["arg2_values"], 1))) if n else
          And(And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg2_length"] == 2), Select(v["arg2_values"], 0) < Select(v["arg2_values"], 1)))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 14
        rule_14(solver, {'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
