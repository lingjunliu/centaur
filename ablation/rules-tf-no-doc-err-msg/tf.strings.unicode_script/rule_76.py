import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If tensor v_1 has dtype string, its min value must be inside the provided list (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == str, Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 0) == 1), Select(v["arg1_range"], 0) == 2), Select(v["arg1_range"], 0) == 3), Select(v["arg1_range"], 0) == 4), Select(v["arg1_range"], 0) == 5), Select(v["arg1_range"], 0) == 6), Select(v["arg1_range"], 0) == 7), Select(v["arg1_range"], 0) == 8), Select(v["arg1_range"], 0) == 9), Select(v["arg1_range"], 0) == 10), Select(v["arg1_range"], 0) == 11), Select(v["arg1_range"], 0) == 12), Select(v["arg1_range"], 0) == 13), Select(v["arg1_range"], 0) == 14), Select(v["arg1_range"], 0) == 15), Select(v["arg1_range"], 0) == 16), Select(v["arg1_range"], 0) == 17), Select(v["arg1_range"], 0) == 18), Select(v["arg1_range"], 0) == 19), Select(v["arg1_range"], 0) == 20), Select(v["arg1_range"], 0) == 21), Select(v["arg1_range"], 0) == 22), Select(v["arg1_range"], 0) == 23), Select(v["arg1_range"], 0) == 24), Select(v["arg1_range"], 0) == 25), True)) if n else
          If(v["arg1_dtype"] == str, Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 0) == 1), Select(v["arg1_range"], 0) == 2), Select(v["arg1_range"], 0) == 3), Select(v["arg1_range"], 0) == 4), Select(v["arg1_range"], 0) == 5), Select(v["arg1_range"], 0) == 6), Select(v["arg1_range"], 0) == 7), Select(v["arg1_range"], 0) == 8), Select(v["arg1_range"], 0) == 9), Select(v["arg1_range"], 0) == 10), Select(v["arg1_range"], 0) == 11), Select(v["arg1_range"], 0) == 12), Select(v["arg1_range"], 0) == 13), Select(v["arg1_range"], 0) == 14), Select(v["arg1_range"], 0) == 15), Select(v["arg1_range"], 0) == 16), Select(v["arg1_range"], 0) == 17), Select(v["arg1_range"], 0) == 18), Select(v["arg1_range"], 0) == 19), Select(v["arg1_range"], 0) == 20), Select(v["arg1_range"], 0) == 21), Select(v["arg1_range"], 0) == 22), Select(v["arg1_range"], 0) == 23), Select(v["arg1_range"], 0) == 24), Select(v["arg1_range"], 0) == 25), True))
)

def rule_76_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 76
        rule_76(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)
