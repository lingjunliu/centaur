import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype of var is int32 or int64, then lr, l1, l2 and grad must also be of type int32 or int64 (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4), And(And(And(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4)), Or(v["arg4_dtype"] == 3, v["arg4_dtype"] == 4)), Or(v["arg5_dtype"] == 3, v["arg5_dtype"] == 4)), True)) if n else
          If(Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4), And(And(And(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4)), Or(v["arg4_dtype"] == 3, v["arg4_dtype"] == 4)), Or(v["arg5_dtype"] == 3, v["arg5_dtype"] == 4)), True))
)

def rule_85_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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

        # Constraints for rule 85
        rule_85(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype']}, neg)
