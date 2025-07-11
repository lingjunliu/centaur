import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Data types of var, m, lr, alpha, sign_decay, beta and grad must be one of the allowed types (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 6), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 14), v["arg1_dtype"] == 10), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18), v["arg1_dtype"] == 5), v["arg1_dtype"] == 15)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 3), v["arg2_dtype"] == 6), v["arg2_dtype"] == 2), v["arg2_dtype"] == 1), v["arg2_dtype"] == 9), v["arg2_dtype"] == 4), v["arg2_dtype"] == 14), v["arg2_dtype"] == 10), v["arg2_dtype"] == 16), v["arg2_dtype"] == 17), v["arg2_dtype"] == 18), v["arg2_dtype"] == 5), v["arg2_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 3), v["arg3_dtype"] == 6), v["arg3_dtype"] == 2), v["arg3_dtype"] == 1), v["arg3_dtype"] == 9), v["arg3_dtype"] == 4), v["arg3_dtype"] == 14), v["arg3_dtype"] == 10), v["arg3_dtype"] == 16), v["arg3_dtype"] == 17), v["arg3_dtype"] == 18), v["arg3_dtype"] == 5), v["arg3_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8), v["arg4_dtype"] == 3), v["arg4_dtype"] == 6), v["arg4_dtype"] == 2), v["arg4_dtype"] == 1), v["arg4_dtype"] == 9), v["arg4_dtype"] == 4), v["arg4_dtype"] == 14), v["arg4_dtype"] == 10), v["arg4_dtype"] == 16), v["arg4_dtype"] == 17), v["arg4_dtype"] == 18), v["arg4_dtype"] == 5), v["arg4_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8), v["arg5_dtype"] == 3), v["arg5_dtype"] == 6), v["arg5_dtype"] == 2), v["arg5_dtype"] == 1), v["arg5_dtype"] == 9), v["arg5_dtype"] == 4), v["arg5_dtype"] == 14), v["arg5_dtype"] == 10), v["arg5_dtype"] == 16), v["arg5_dtype"] == 17), v["arg5_dtype"] == 18), v["arg5_dtype"] == 5), v["arg5_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg6_dtype"] == 7, v["arg6_dtype"] == 8), v["arg6_dtype"] == 3), v["arg6_dtype"] == 6), v["arg6_dtype"] == 2), v["arg6_dtype"] == 1), v["arg6_dtype"] == 9), v["arg6_dtype"] == 4), v["arg6_dtype"] == 14), v["arg6_dtype"] == 10), v["arg6_dtype"] == 16), v["arg6_dtype"] == 17), v["arg6_dtype"] == 18), v["arg6_dtype"] == 5), v["arg6_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg7_dtype"] == 7, v["arg7_dtype"] == 8), v["arg7_dtype"] == 3), v["arg7_dtype"] == 6), v["arg7_dtype"] == 2), v["arg7_dtype"] == 1), v["arg7_dtype"] == 9), v["arg7_dtype"] == 4), v["arg7_dtype"] == 14), v["arg7_dtype"] == 10), v["arg7_dtype"] == 16), v["arg7_dtype"] == 17), v["arg7_dtype"] == 18), v["arg7_dtype"] == 5), v["arg7_dtype"] == 15)))) if n else
          And(And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 6), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 14), v["arg1_dtype"] == 10), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18), v["arg1_dtype"] == 5), v["arg1_dtype"] == 15)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 3), v["arg2_dtype"] == 6), v["arg2_dtype"] == 2), v["arg2_dtype"] == 1), v["arg2_dtype"] == 9), v["arg2_dtype"] == 4), v["arg2_dtype"] == 14), v["arg2_dtype"] == 10), v["arg2_dtype"] == 16), v["arg2_dtype"] == 17), v["arg2_dtype"] == 18), v["arg2_dtype"] == 5), v["arg2_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 3), v["arg3_dtype"] == 6), v["arg3_dtype"] == 2), v["arg3_dtype"] == 1), v["arg3_dtype"] == 9), v["arg3_dtype"] == 4), v["arg3_dtype"] == 14), v["arg3_dtype"] == 10), v["arg3_dtype"] == 16), v["arg3_dtype"] == 17), v["arg3_dtype"] == 18), v["arg3_dtype"] == 5), v["arg3_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8), v["arg4_dtype"] == 3), v["arg4_dtype"] == 6), v["arg4_dtype"] == 2), v["arg4_dtype"] == 1), v["arg4_dtype"] == 9), v["arg4_dtype"] == 4), v["arg4_dtype"] == 14), v["arg4_dtype"] == 10), v["arg4_dtype"] == 16), v["arg4_dtype"] == 17), v["arg4_dtype"] == 18), v["arg4_dtype"] == 5), v["arg4_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8), v["arg5_dtype"] == 3), v["arg5_dtype"] == 6), v["arg5_dtype"] == 2), v["arg5_dtype"] == 1), v["arg5_dtype"] == 9), v["arg5_dtype"] == 4), v["arg5_dtype"] == 14), v["arg5_dtype"] == 10), v["arg5_dtype"] == 16), v["arg5_dtype"] == 17), v["arg5_dtype"] == 18), v["arg5_dtype"] == 5), v["arg5_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg6_dtype"] == 7, v["arg6_dtype"] == 8), v["arg6_dtype"] == 3), v["arg6_dtype"] == 6), v["arg6_dtype"] == 2), v["arg6_dtype"] == 1), v["arg6_dtype"] == 9), v["arg6_dtype"] == 4), v["arg6_dtype"] == 14), v["arg6_dtype"] == 10), v["arg6_dtype"] == 16), v["arg6_dtype"] == 17), v["arg6_dtype"] == 18), v["arg6_dtype"] == 5), v["arg6_dtype"] == 15))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg7_dtype"] == 7, v["arg7_dtype"] == 8), v["arg7_dtype"] == 3), v["arg7_dtype"] == 6), v["arg7_dtype"] == 2), v["arg7_dtype"] == 1), v["arg7_dtype"] == 9), v["arg7_dtype"] == 4), v["arg7_dtype"] == 14), v["arg7_dtype"] == 10), v["arg7_dtype"] == 16), v["arg7_dtype"] == 17), v["arg7_dtype"] == 18), v["arg7_dtype"] == 5), v["arg7_dtype"] == 15))))
)

def rule_12_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

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
        if not isinstance(arg7, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')
        arg6_dtype = Int('arg6_dtype')
        arg7_dtype = Int('arg7_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))
        solver.add(arg6_dtype == list_of_available_dtypes.index(arg6.dtype))
        solver.add(arg7_dtype == list_of_available_dtypes.index(arg7.dtype))

        # Constraints for rule 12
        rule_12(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype, 'arg6_dtype': arg6_dtype, 'arg7_dtype': arg7_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype'], 'arg6_dtype': arg6['dtype'], 'arg7_dtype': arg7['dtype']}, neg)
