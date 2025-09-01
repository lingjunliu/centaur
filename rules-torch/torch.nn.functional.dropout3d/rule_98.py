import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dtype should be a floating point and if training is true, probability p must be between 0 and 1 exclusive, otherwise, if training is false, probability p must be zero (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), (If(v["arg2_value"], And(v["arg3_value"] > 0, v["arg3_value"] < 1), v["arg3_value"] == 0)))) if n else
          And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), (If(v["arg2_value"], And(v["arg3_value"] > 0, v["arg3_value"] < 1), v["arg3_value"] == 0))))
)

def rule_98_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 98
        rule_98(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
