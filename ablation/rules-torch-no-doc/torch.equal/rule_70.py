import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if tensors share elements, check same dtype (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If((Or([And(i < (Select(v["arg1_range"], 1) + 1), Or([And(j < (Select(v["arg2_range"], 1) + 1), i == j) for j in range(6)])) for i in range(6)])), (v["arg1_dtype"] == v["arg2_dtype"]), True)) if n else
          If((Or([And(i < (Select(v["arg1_range"], 1) + 1), Or([And(j < (Select(v["arg2_range"], 1) + 1), i == j) for j in range(6)])) for i in range(6)])), (v["arg1_dtype"] == v["arg2_dtype"]), True))
)

def rule_70_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']}, neg)
