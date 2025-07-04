import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Lower and upper values should be in the range [-1, 1] if training = True and if dtype is float16 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] == True, v["arg3_dtype"] == 6), And(And(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1), v["arg2_value"] >= -1), v["arg2_value"] <= 1), False)) if n else
          If(And(v["arg4_value"] == True, v["arg3_dtype"] == 6), And(And(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1), v["arg2_value"] >= -1), v["arg2_value"] <= 1), False))
)

def rule_20_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
