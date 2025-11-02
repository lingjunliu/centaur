import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

#  value and threshold must be in the same range for the type of the input tensor, if this is an integer one. (Rule 143)

rule_143 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_dtype"] == 1), And((And(v["arg2_value"] >= -128, v["arg2_value"] <= 127)), (And(v["arg3_value"] >= -128, v["arg3_value"] <= 127))), If((v["arg1_dtype"] == 2), And((And(v["arg2_value"] >= -32768, v["arg2_value"] <= 32767)), (And(v["arg3_value"] >= -32768, v["arg3_value"] <= 32767))), If((v["arg1_dtype"] == 3), And((And(v["arg2_value"] >= -2147483648, v["arg2_value"] <= 2147483647)), (And(v["arg3_value"] >= -2147483648, v["arg3_value"] <= 2147483647))), True)))) if n else
          If((v["arg1_dtype"] == 1), And((And(v["arg2_value"] >= -128, v["arg2_value"] <= 127)), (And(v["arg3_value"] >= -128, v["arg3_value"] <= 127))), If((v["arg1_dtype"] == 2), And((And(v["arg2_value"] >= -32768, v["arg2_value"] <= 32767)), (And(v["arg3_value"] >= -32768, v["arg3_value"] <= 32767))), If((v["arg1_dtype"] == 3), And((And(v["arg2_value"] >= -2147483648, v["arg2_value"] <= 2147483647)), (And(v["arg3_value"] >= -2147483648, v["arg3_value"] <= 2147483647))), True))))
)

def rule_143_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 143
        rule_143(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_143(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
