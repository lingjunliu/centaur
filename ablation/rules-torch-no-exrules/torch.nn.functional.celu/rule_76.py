import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input has ndim greater than zero, then alpha must be specified and not be zero and less than 1000 and alpha not equal to 1 (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, And(And(And((And(v["arg2_value"] > -1.0E+10, v["arg2_value"] < 1.0E+10)), v["arg2_value"] != 0), v["arg2_value"] < 1000), v["arg2_value"] != 1), True)) if n else
          If(v["arg1_ndim"] > 0, And(And(And((And(v["arg2_value"] > -1.0E+10, v["arg2_value"] < 1.0E+10)), v["arg2_value"] != 0), v["arg2_value"] < 1000), v["arg2_value"] != 1), True))
)

def rule_76_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)

        # Constraints for rule 76
        rule_76(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
