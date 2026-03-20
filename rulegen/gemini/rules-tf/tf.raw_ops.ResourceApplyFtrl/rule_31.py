import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, and linear cannot be int8, uint8, int16, uint16, int32, uint32, int64, or uint64 if grad is float (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 7, And(And((And(And(And(And(And(And(v["arg2_dtype"] != 1, v["arg2_dtype"] != 5), v["arg2_dtype"] != 2), v["arg2_dtype"] != 16), v["arg2_dtype"] != 3), v["arg2_dtype"] != 17), v["arg2_dtype"] != 4)), (And(And(And(And(And(And(v["arg3_dtype"] != 1, v["arg3_dtype"] != 5), v["arg3_dtype"] != 2), v["arg3_dtype"] != 16), v["arg3_dtype"] != 3), v["arg3_dtype"] != 17), v["arg3_dtype"] != 4))), (And(And(And(And(And(And(v["arg4_dtype"] != 1, v["arg4_dtype"] != 5), v["arg4_dtype"] != 2), v["arg4_dtype"] != 16), v["arg4_dtype"] != 3), v["arg4_dtype"] != 17), v["arg4_dtype"] != 4))), True)) if n else
          If(v["arg1_dtype"] == 7, And(And((And(And(And(And(And(And(v["arg2_dtype"] != 1, v["arg2_dtype"] != 5), v["arg2_dtype"] != 2), v["arg2_dtype"] != 16), v["arg2_dtype"] != 3), v["arg2_dtype"] != 17), v["arg2_dtype"] != 4)), (And(And(And(And(And(And(v["arg3_dtype"] != 1, v["arg3_dtype"] != 5), v["arg3_dtype"] != 2), v["arg3_dtype"] != 16), v["arg3_dtype"] != 3), v["arg3_dtype"] != 17), v["arg3_dtype"] != 4))), (And(And(And(And(And(And(v["arg4_dtype"] != 1, v["arg4_dtype"] != 5), v["arg4_dtype"] != 2), v["arg4_dtype"] != 16), v["arg4_dtype"] != 3), v["arg4_dtype"] != 17), v["arg4_dtype"] != 4))), True))
)

def rule_31_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

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

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 31
        rule_31(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype']}, neg)
