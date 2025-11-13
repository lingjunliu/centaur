import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If var is bfloat16 or half, then lr, l1, l2, lr_power tensors should be float or bfloat16/half accordingly (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_dtype"] == 6), And(And(And((Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8)), (Or(Or(v["arg3_dtype"] == 6, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8))), (Or(Or(v["arg4_dtype"] == 6, v["arg4_dtype"] == 7), v["arg4_dtype"] == 8))), (Or(Or(v["arg5_dtype"] == 6, v["arg5_dtype"] == 7), v["arg5_dtype"] == 8))), If((v["arg1_dtype"] == 16), And(And(And((Or(Or(v["arg2_dtype"] == 16, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8)), (Or(Or(v["arg3_dtype"] == 16, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8))), (Or(Or(v["arg4_dtype"] == 16, v["arg4_dtype"] == 7), v["arg4_dtype"] == 8))), (Or(Or(v["arg5_dtype"] == 16, v["arg5_dtype"] == 7), v["arg5_dtype"] == 8))), True))) if n else
          If((v["arg1_dtype"] == 6), And(And(And((Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8)), (Or(Or(v["arg3_dtype"] == 6, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8))), (Or(Or(v["arg4_dtype"] == 6, v["arg4_dtype"] == 7), v["arg4_dtype"] == 8))), (Or(Or(v["arg5_dtype"] == 6, v["arg5_dtype"] == 7), v["arg5_dtype"] == 8))), If((v["arg1_dtype"] == 16), And(And(And((Or(Or(v["arg2_dtype"] == 16, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8)), (Or(Or(v["arg3_dtype"] == 16, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8))), (Or(Or(v["arg4_dtype"] == 16, v["arg4_dtype"] == 7), v["arg4_dtype"] == 8))), (Or(Or(v["arg5_dtype"] == 16, v["arg5_dtype"] == 7), v["arg5_dtype"] == 8))), True)))
)

def rule_20_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

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

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))

        # Constraints for rule 20
        rule_20(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype']}, neg)
