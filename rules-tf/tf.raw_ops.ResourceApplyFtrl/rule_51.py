import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If lr_power is not 0, then l1 and l2 must be positive and grad's type cannot be integer (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) != 0, And(And(Select(v["arg2_range"], 0) > 0, Select(v["arg3_range"], 0) > 0), v["arg4_dtype"] > 6), True)) if n else
          If(Select(v["arg1_range"], 0) != 0, And(And(Select(v["arg2_range"], 0) > 0, Select(v["arg3_range"], 0) > 0), v["arg4_dtype"] > 6), True))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 51
        rule_51(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_range': arg3_range, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range'], 'arg4_dtype': arg4['dtype']}, neg)
