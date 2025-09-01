import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if other is not a tensor, then other has to be small, check against max value of float16 to avoid overflow, check when other is float and alpha is not too small or not too large (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_dtype"] < 6, And(And(And(v["arg1_value"] < 65500, v["arg1_value"] > -65500), v["arg2_value"] < 1000), v["arg2_value"] > -1000), True)) if n else
          If(v["arg3_dtype"] < 6, And(And(And(v["arg1_value"] < 65500, v["arg1_value"] > -65500), v["arg2_value"] < 1000), v["arg2_value"] > -1000), True))
)

def rule_100_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 100
        rule_100(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
