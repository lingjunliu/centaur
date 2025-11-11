import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# half-precision safe magnitudes for alpha, beta, and k to avoid casting overflow (Rule 7)

rule_7 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_dtype"] == 6, And(And(And(And(And(v["arg1_value"] <= 65504, v["arg1_value"] >= -65504), v["arg2_value"] <= 65504), v["arg2_value"] >= -65504), v["arg3_value"] <= 65504), v["arg3_value"] >= -65504), True)) if n else
          If(v["arg4_dtype"] == 6, And(And(And(And(And(v["arg1_value"] <= 65504, v["arg1_value"] >= -65504), v["arg2_value"] <= 65504), v["arg2_value"] >= -65504), v["arg3_value"] <= 65504), v["arg3_value"] >= -65504), True))
)

def rule_7_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 7
        rule_7(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_7(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype']}, neg)
