import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If size is an int and requires_grad = True, and dtype not int, then it must be valid for the current device (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == True, (Or(Or(Or(Or(v["arg3_value"] == 6, v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10))), v["arg1_value"] > 0, True)) if n else
          If(And(v["arg2_value"] == True, (Or(Or(Or(Or(v["arg3_value"] == 6, v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10))), v["arg1_value"] > 0, True))
)

def rule_68_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
