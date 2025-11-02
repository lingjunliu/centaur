import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If training is true, probability p should be between 0 and 1 inclusive and greater than 0, and inplace influences dtype (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], And((And(0.0 < v["arg1_value"], v["arg1_value"] < 1.0)), If(v["arg4_value"], Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 6), True)), True)) if n else
          If(v["arg2_value"], And((And(0.0 < v["arg1_value"], v["arg1_value"] < 1.0)), If(v["arg4_value"], Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 6), True)), True))
)

def rule_65_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
