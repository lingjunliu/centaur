import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, accum_update, lr, rho, epsilon and grad can only be from a limited set of types (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 2), v["arg1_dtype"] == 5), v["arg1_dtype"] == 3), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18), v["arg1_dtype"] == 20), v["arg1_dtype"] == 19), v["arg1_dtype"] == 21), v["arg1_dtype"] == 22)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), v["arg1_dtype"] == v["arg5_dtype"]), v["arg1_dtype"] == v["arg6_dtype"]), v["arg1_dtype"] == v["arg7_dtype"])) if n else
          And(And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 2), v["arg1_dtype"] == 5), v["arg1_dtype"] == 3), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18), v["arg1_dtype"] == 20), v["arg1_dtype"] == 19), v["arg1_dtype"] == 21), v["arg1_dtype"] == 22)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), v["arg1_dtype"] == v["arg5_dtype"]), v["arg1_dtype"] == v["arg6_dtype"]), v["arg1_dtype"] == v["arg7_dtype"]))
)

def rule_31_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
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

        # Constraints for rule 31
        rule_31(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype, 'arg6_dtype': arg6_dtype, 'arg7_dtype': arg7_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype'], 'arg6_dtype': arg6['dtype'], 'arg7_dtype': arg7['dtype']}, neg)
