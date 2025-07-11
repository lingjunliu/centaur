import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If single element tensor's dtype is string, its min/max can be only be selected from predefined list (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, And(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 0) == 1), Select(v["arg1_range"], 0) == 2), Select(v["arg1_range"], 0) == 3), Select(v["arg1_range"], 0) == 4), Select(v["arg1_range"], 0) == 5), Select(v["arg1_range"], 0) == 6), Select(v["arg1_range"], 0) == 7), Select(v["arg1_range"], 0) == 8), Select(v["arg1_range"], 0) == 9), Select(v["arg1_range"], 0) == 10), Select(v["arg1_range"], 0) == 11), Select(v["arg1_range"], 0) == 12), Select(v["arg1_range"], 0) == 13), Select(v["arg1_range"], 0) == 14), Select(v["arg1_range"], 0) == 15), Select(v["arg1_range"], 0) == 16), Select(v["arg1_range"], 0) == 17), Select(v["arg1_range"], 0) == 18), Select(v["arg1_range"], 0) == 19), Select(v["arg1_range"], 0) == 20), Select(v["arg1_range"], 0) == 21), Select(v["arg1_range"], 0) == 22), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 1) == 0, Select(v["arg1_range"], 1) == 1), Select(v["arg1_range"], 1) == 2), Select(v["arg1_range"], 1) == 3), Select(v["arg1_range"], 1) == 4), Select(v["arg1_range"], 1) == 5), Select(v["arg1_range"], 1) == 6), Select(v["arg1_range"], 1) == 7), Select(v["arg1_range"], 1) == 8), Select(v["arg1_range"], 1) == 9), Select(v["arg1_range"], 1) == 10), Select(v["arg1_range"], 1) == 11), Select(v["arg1_range"], 1) == 12), Select(v["arg1_range"], 1) == 13), Select(v["arg1_range"], 1) == 14), Select(v["arg1_range"], 1) == 15), Select(v["arg1_range"], 1) == 16), Select(v["arg1_range"], 1) == 17), Select(v["arg1_range"], 1) == 18), Select(v["arg1_range"], 1) == 19), Select(v["arg1_range"], 1) == 20), Select(v["arg1_range"], 1) == 21), Select(v["arg1_range"], 1) == 22)), False)) if n else
          If(v["arg1_dtype"] == 11, And(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 0) == 1), Select(v["arg1_range"], 0) == 2), Select(v["arg1_range"], 0) == 3), Select(v["arg1_range"], 0) == 4), Select(v["arg1_range"], 0) == 5), Select(v["arg1_range"], 0) == 6), Select(v["arg1_range"], 0) == 7), Select(v["arg1_range"], 0) == 8), Select(v["arg1_range"], 0) == 9), Select(v["arg1_range"], 0) == 10), Select(v["arg1_range"], 0) == 11), Select(v["arg1_range"], 0) == 12), Select(v["arg1_range"], 0) == 13), Select(v["arg1_range"], 0) == 14), Select(v["arg1_range"], 0) == 15), Select(v["arg1_range"], 0) == 16), Select(v["arg1_range"], 0) == 17), Select(v["arg1_range"], 0) == 18), Select(v["arg1_range"], 0) == 19), Select(v["arg1_range"], 0) == 20), Select(v["arg1_range"], 0) == 21), Select(v["arg1_range"], 0) == 22), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 1) == 0, Select(v["arg1_range"], 1) == 1), Select(v["arg1_range"], 1) == 2), Select(v["arg1_range"], 1) == 3), Select(v["arg1_range"], 1) == 4), Select(v["arg1_range"], 1) == 5), Select(v["arg1_range"], 1) == 6), Select(v["arg1_range"], 1) == 7), Select(v["arg1_range"], 1) == 8), Select(v["arg1_range"], 1) == 9), Select(v["arg1_range"], 1) == 10), Select(v["arg1_range"], 1) == 11), Select(v["arg1_range"], 1) == 12), Select(v["arg1_range"], 1) == 13), Select(v["arg1_range"], 1) == 14), Select(v["arg1_range"], 1) == 15), Select(v["arg1_range"], 1) == 16), Select(v["arg1_range"], 1) == 17), Select(v["arg1_range"], 1) == 18), Select(v["arg1_range"], 1) == 19), Select(v["arg1_range"], 1) == 20), Select(v["arg1_range"], 1) == 21), Select(v["arg1_range"], 1) == 22)), False))
)

def rule_19_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 19
        rule_19(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)
