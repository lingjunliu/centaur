import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If the out tensor is provided, and the input tensor is float or complex, and its minimum value is non-negative, then the out tensor must be float or complex with at least equal precision (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10))), (Select(v["arg1_range"], 0) >= 0)), If(v["arg1_dtype"] == 7, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 8, Or(Or(v["arg2_dtype"] == 8, v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 9, Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 10, v["arg2_dtype"] == 10, False)))), False)) if n else
          If(And((Or(Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10))), (Select(v["arg1_range"], 0) >= 0)), If(v["arg1_dtype"] == 7, Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 8, Or(Or(v["arg2_dtype"] == 8, v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 9, Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), If(v["arg1_dtype"] == 10, v["arg2_dtype"] == 10, False)))), False))
)

def rule_47_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 47
        rule_47(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)
