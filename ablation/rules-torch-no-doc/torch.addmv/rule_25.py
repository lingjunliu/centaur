import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If beta is not 1 and alpha is not 1, the out tensor dtype should be same as mat and vec, if out is not specified, mat and vec dtype should be float (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg4_value"] != 1, v["arg5_value"] != 1)), And(v["arg1_dtype"] == v["arg2_dtype"], v["arg2_dtype"] == v["arg3_dtype"]), If(v["arg2_dtype"] == 8, True, If(v["arg3_dtype"] == 8, True, False)))) if n else
          If((Or(v["arg4_value"] != 1, v["arg5_value"] != 1)), And(v["arg1_dtype"] == v["arg2_dtype"], v["arg2_dtype"] == v["arg3_dtype"]), If(v["arg2_dtype"] == 8, True, If(v["arg3_dtype"] == 8, True, False))))
)

def rule_25_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not ((isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)) or isinstance(arg4, (float, np.floating))):
            return False
        if not ((isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)) or isinstance(arg5, (float, np.floating))):
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

        # Constraints for rule 25
        rule_25(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
