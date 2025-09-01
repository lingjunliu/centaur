import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Row and column must be non-negative, their product should fit into int64 to prevent overflows on CUDA, and dtype should be int64 (torch.long (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] >= 0, v["arg2_value"] >= 0), (If(v["arg1_value"] > 0, v["arg2_value"] <= 9223372036854775807 / v["arg1_value"], True))), v["arg3_value"] == 4)) if n else
          And(And(And(v["arg1_value"] >= 0, v["arg2_value"] >= 0), (If(v["arg1_value"] > 0, v["arg2_value"] <= 9223372036854775807 / v["arg1_value"], True))), v["arg3_value"] == 4))
)

def rule_72_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
