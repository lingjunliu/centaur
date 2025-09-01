import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If an out tensor is supplied, then it needs to be a data type that can store a float if the input is a float. (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_dtype"] == 6), Or(Or(Or(Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 9)), (v["arg2_dtype"] == 10)), (v["arg2_dtype"] == 11)), If((v["arg1_dtype"] == 7), Or(Or(Or(Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 9)), (v["arg2_dtype"] == 10)), (v["arg2_dtype"] == 11)), If((v["arg1_dtype"] == 8), Or(Or(Or((v["arg2_dtype"] == 8), (v["arg2_dtype"] == 9)), (v["arg2_dtype"] == 10)), (v["arg2_dtype"] == 11)), True)))) if n else
          If((v["arg1_dtype"] == 6), Or(Or(Or(Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 9)), (v["arg2_dtype"] == 10)), (v["arg2_dtype"] == 11)), If((v["arg1_dtype"] == 7), Or(Or(Or(Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 9)), (v["arg2_dtype"] == 10)), (v["arg2_dtype"] == 11)), If((v["arg1_dtype"] == 8), Or(Or(Or((v["arg2_dtype"] == 8), (v["arg2_dtype"] == 9)), (v["arg2_dtype"] == 10)), (v["arg2_dtype"] == 11)), True))))
)

def rule_74_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 74
        rule_74(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
