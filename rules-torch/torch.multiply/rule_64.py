import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If v_1 is complex and v_2 is not complex then v_3 (out (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11)), (And((v["arg2_dtype"] != 10), (v["arg2_dtype"] != 11)))), (Or((Or(v["arg3_dtype"] == 10, v["arg3_dtype"] == 11)), (v["arg3_dtype"] == v["arg2_dtype"]))), False)) if n else
          If(And((Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11)), (And((v["arg2_dtype"] != 10), (v["arg2_dtype"] != 11)))), (Or((Or(v["arg3_dtype"] == 10, v["arg3_dtype"] == 11)), (v["arg3_dtype"] == v["arg2_dtype"]))), False))
)

def rule_64_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 64
        rule_64(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
