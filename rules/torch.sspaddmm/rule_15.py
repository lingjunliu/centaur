import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if alpha or beta not equal to 1.0, input and mat1 must have same dtype as result of mat1 x mat2 (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == 1, v["arg4_value"] == 1), True, v["arg1_dtype"] == v["arg2_dtype"])) if n else
          If(And(v["arg3_value"] == 1, v["arg4_value"] == 1), True, v["arg1_dtype"] == v["arg2_dtype"]))
)

def rule_15_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, (float, np.floating)) or (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False
        if not (isinstance(arg4, (float, np.floating)) or (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 15
        rule_15(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
