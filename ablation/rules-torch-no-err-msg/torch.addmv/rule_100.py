import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input, mat and vec are of type FloatTensor or DoubleTensor, beta and alpha must be real numbers (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), (Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8)))) if n else
          And(And((Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), (Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))))
)

def rule_100_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 100
        rule_100(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
