import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If an output tensor is provided, and both input and output are not dtype, and input tensor is floating point, its dtype must also be floating point (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg2_dtype"] != 12, v["arg1_dtype"] != 12)), If((Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9)), (Or(v["arg2_dtype"] == 8, v["arg2_dtype"] == 9)), True), True)) if n else
          If((And(v["arg2_dtype"] != 12, v["arg1_dtype"] != 12)), If((Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9)), (Or(v["arg2_dtype"] == 8, v["arg2_dtype"] == 9)), True), True))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
