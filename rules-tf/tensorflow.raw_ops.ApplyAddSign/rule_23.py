import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If var is float16 or half, then lr, alpha, sign_decay, beta and grad must be float16 or half (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 16), And(And(And(And((Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 16)), (Or(v["arg3_dtype"] == 6, v["arg3_dtype"] == 16))), (Or(v["arg4_dtype"] == 6, v["arg4_dtype"] == 16))), (Or(v["arg5_dtype"] == 6, v["arg5_dtype"] == 16))), (Or(v["arg6_dtype"] == 6, v["arg6_dtype"] == 16))), False)) if n else
          If(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 16), And(And(And(And((Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 16)), (Or(v["arg3_dtype"] == 6, v["arg3_dtype"] == 16))), (Or(v["arg4_dtype"] == 6, v["arg4_dtype"] == 16))), (Or(v["arg5_dtype"] == 6, v["arg5_dtype"] == 16))), (Or(v["arg6_dtype"] == 6, v["arg6_dtype"] == 16))), False))
)

def rule_23_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

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
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')
        arg6_dtype = Int('arg6_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))
        solver.add(arg6_dtype == list_of_available_dtypes.index(arg6.dtype))

        # Constraints for rule 23
        rule_23(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype, 'arg6_dtype': arg6_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype'], 'arg6_dtype': arg6['dtype']}, neg)
