import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If input_tensor has shape (2,3,4 (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(v["arg1_ndim"] == 3, v["arg2_length"] == 2), Select(v["arg2_values"], 0) == 0), Select(v["arg2_values"], 1) == 2), v["arg3_value"] == False), v["arg1_ndim"] - v["arg2_length"] == 1, False)) if n else
          If(And(And(And(And(v["arg1_ndim"] == 3, v["arg2_length"] == 2), Select(v["arg2_values"], 0) == 0), Select(v["arg2_values"], 1) == 2), v["arg3_value"] == False), v["arg1_ndim"] - v["arg2_length"] == 1, False))
)

def rule_14_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 14
        rule_14(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
