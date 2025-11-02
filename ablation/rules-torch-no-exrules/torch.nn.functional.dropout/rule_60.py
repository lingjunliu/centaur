import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Relate training and inplace. If inplace=true, training must also be true, relate to the tensor and probability (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], And(And(v["arg1_value"], (Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 6))), (And(0.0 <= v["arg4_value"], v["arg4_value"] <= 1.0))), True)) if n else
          If(v["arg2_value"], And(And(v["arg1_value"], (Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 6))), (And(0.0 <= v["arg4_value"], v["arg4_value"] <= 1.0))), True))
)

def rule_60_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
