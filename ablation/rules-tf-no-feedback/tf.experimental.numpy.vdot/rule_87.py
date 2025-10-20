import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Make sure tensors have same sign of the min and max. (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) < 0, And(Select(v["arg2_range"], 0) < 0, If(Select(v["arg1_range"], 1) > 0, Select(v["arg2_range"], 1) > 0, True)), True)) if n else
          If(Select(v["arg1_range"], 0) < 0, And(Select(v["arg2_range"], 0) < 0, If(Select(v["arg1_range"], 1) > 0, Select(v["arg2_range"], 1) > 0, True)), True))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 87
        rule_87(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range']}, neg)
