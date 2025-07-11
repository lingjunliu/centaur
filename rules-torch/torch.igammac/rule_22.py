import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If input's dtype is integer, then other's dtype must be integer or float, and the dtypes must be compatible. (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_dtype"] >= 1), (v["arg1_dtype"] <= 5)), And((And((v["arg2_dtype"] >= 1), (v["arg2_dtype"] <= 8))), (Or(Or(Or(Or((And(v["arg1_dtype"] == 1, v["arg2_dtype"] == 1)), (And(v["arg1_dtype"] == 2, v["arg2_dtype"] == 2))), (And(v["arg1_dtype"] == 3, v["arg2_dtype"] == 3))), (And(v["arg1_dtype"] == 4, v["arg2_dtype"] == 4))), (And(v["arg1_dtype"] == 5, v["arg2_dtype"] == 5))))), False)) if n else
          If(And((v["arg1_dtype"] >= 1), (v["arg1_dtype"] <= 5)), And((And((v["arg2_dtype"] >= 1), (v["arg2_dtype"] <= 8))), (Or(Or(Or(Or((And(v["arg1_dtype"] == 1, v["arg2_dtype"] == 1)), (And(v["arg1_dtype"] == 2, v["arg2_dtype"] == 2))), (And(v["arg1_dtype"] == 3, v["arg2_dtype"] == 3))), (And(v["arg1_dtype"] == 4, v["arg2_dtype"] == 4))), (And(v["arg1_dtype"] == 5, v["arg2_dtype"] == 5))))), False))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 22
        rule_22(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
