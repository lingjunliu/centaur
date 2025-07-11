import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input min must not be greater than the maximum possible value for the input dtype (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, Select(v["arg2_range"], 0) <= 127, If(v["arg1_dtype"] == 5, Select(v["arg2_range"], 0) <= 255, If(v["arg1_dtype"] == 3, Select(v["arg2_range"], 0) <= 2147483647, If(v["arg1_dtype"] == 2, Select(v["arg2_range"], 0) <= 32767, If(v["arg1_dtype"] == 13, Select(v["arg2_range"], 0) <= 65535, False)))))) if n else
          If(v["arg1_dtype"] == 1, Select(v["arg2_range"], 0) <= 127, If(v["arg1_dtype"] == 5, Select(v["arg2_range"], 0) <= 255, If(v["arg1_dtype"] == 3, Select(v["arg2_range"], 0) <= 2147483647, If(v["arg1_dtype"] == 2, Select(v["arg2_range"], 0) <= 32767, If(v["arg1_dtype"] == 13, Select(v["arg2_range"], 0) <= 65535, False))))))
)

def rule_18_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 18
        rule_18(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range']}, neg)
