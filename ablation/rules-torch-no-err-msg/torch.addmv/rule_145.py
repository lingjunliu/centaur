import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# For alpha and beta to apply correct calculation must follow below properties (Rule 145)

rule_145 = lambda s, v, n=False: (
    s.add(Not((If(And(And((Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8)), (Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8))), (Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8))), And((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), And((Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5)), (Or(Or(Or(Or(v["arg2_value"] == 1, v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5)))))) if n else
          (If(And(And((Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8)), (Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8))), (Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8))), And((Or(v["arg1_value"] == 7, v["arg1_value"] == 8)), (Or(v["arg2_value"] == 7, v["arg2_value"] == 8))), And((Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5)), (Or(Or(Or(Or(v["arg2_value"] == 1, v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5))))))
)

def rule_145_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, (float, np.floating)) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')

        # Value assignments
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))

        # Constraints for rule 145
        rule_145(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_145(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype']}, neg)
