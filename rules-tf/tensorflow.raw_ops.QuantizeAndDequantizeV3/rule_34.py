import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# num_bits should be less or equal to the number of bits used by the datatype (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 6, Select(v["arg1_range"], 1) <= 15, If(v["arg2_dtype"] == 7, Select(v["arg1_range"], 1) <= 31, If(v["arg2_dtype"] == 8, Select(v["arg1_range"], 1) <= 63, False)))) if n else
          If(v["arg2_dtype"] == 6, Select(v["arg1_range"], 1) <= 15, If(v["arg2_dtype"] == 7, Select(v["arg1_range"], 1) <= 31, If(v["arg2_dtype"] == 8, Select(v["arg1_range"], 1) <= 63, False))))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 34
        rule_34(solver, {'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)
