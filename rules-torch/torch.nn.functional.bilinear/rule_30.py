import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# input tensors, weight, and bias should all have dtypes that are within the same family: integer, float, complex to avoid mixing issues. (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(And(And(v["arg1_dtype"] <= 5, v["arg2_dtype"] <= 5), v["arg3_dtype"] <= 5), v["arg4_dtype"] <= 5)), (And(And(And(And(And(And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), 6 <= v["arg2_dtype"]), v["arg2_dtype"] <= 8), 6 <= v["arg3_dtype"]), v["arg3_dtype"] <= 8), 6 <= v["arg4_dtype"]), v["arg4_dtype"] <= 8))), (And(And(And(v["arg1_dtype"] >= 9, v["arg2_dtype"] >= 9), v["arg3_dtype"] >= 9), v["arg4_dtype"] >= 9)))) if n else
          Or(Or((And(And(And(v["arg1_dtype"] <= 5, v["arg2_dtype"] <= 5), v["arg3_dtype"] <= 5), v["arg4_dtype"] <= 5)), (And(And(And(And(And(And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), 6 <= v["arg2_dtype"]), v["arg2_dtype"] <= 8), 6 <= v["arg3_dtype"]), v["arg3_dtype"] <= 8), 6 <= v["arg4_dtype"]), v["arg4_dtype"] <= 8))), (And(And(And(v["arg1_dtype"] >= 9, v["arg2_dtype"] >= 9), v["arg3_dtype"] >= 9), v["arg4_dtype"] >= 9))))
)

def rule_30_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 30
        rule_30(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype']}, neg)
