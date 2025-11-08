import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input tensor has floating point dtype, and the output dtype is specified as float, then the scale and zero_point should also have floating point dtype (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_dtype"] >= 7, v["arg1_dtype"] <= 8), v["arg2_value"] >= 7), v["arg2_value"] <= 8), (And(And(And(v["arg3_dtype"] >= 7, v["arg3_dtype"] <= 8), v["arg4_dtype"] >= 7), v["arg4_dtype"] <= 8)), True)) if n else
          If(And(And(And(v["arg1_dtype"] >= 7, v["arg1_dtype"] <= 8), v["arg2_value"] >= 7), v["arg2_value"] <= 8), (And(And(And(v["arg3_dtype"] >= 7, v["arg3_dtype"] <= 8), v["arg4_dtype"] >= 7), v["arg4_dtype"] <= 8)), True))
)

def rule_81_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 81
        rule_81(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype']}, neg)
