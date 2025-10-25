import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Tolerances must be non-negative and tensors are float or complex (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg1_value"] >= 0), (v["arg2_value"] >= 0)), (Or(Or(Or(Or(Or(Or(Or(Or(Or((v["arg3_dtype"] == 7), (v["arg3_dtype"] == 8)), (v["arg3_dtype"] == 6)), (v["arg3_dtype"] == 9)), (v["arg3_dtype"] == 10)), (v["arg4_dtype"] == 7)), (v["arg4_dtype"] == 8)), (v["arg4_dtype"] == 6)), (v["arg4_dtype"] == 9)), (v["arg4_dtype"] == 10))))) if n else
          And(And((v["arg1_value"] >= 0), (v["arg2_value"] >= 0)), (Or(Or(Or(Or(Or(Or(Or(Or(Or((v["arg3_dtype"] == 7), (v["arg3_dtype"] == 8)), (v["arg3_dtype"] == 6)), (v["arg3_dtype"] == 9)), (v["arg3_dtype"] == 10)), (v["arg4_dtype"] == 7)), (v["arg4_dtype"] == 8)), (v["arg4_dtype"] == 6)), (v["arg4_dtype"] == 9)), (v["arg4_dtype"] == 10)))))
)

def rule_68_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype']}, neg)
