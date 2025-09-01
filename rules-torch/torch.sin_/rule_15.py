import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Prevent float cast error by requiring data to be a floating point or complex number, or int in a representable range (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_dtype"] > 5), (And(And(v["arg1_dtype"] < 6, Select(v["arg1_range"], 0) > -1.0e+38), Select(v["arg1_range"], 1) < 1.0e+38)))) if n else
          Or((v["arg1_dtype"] > 5), (And(And(v["arg1_dtype"] < 6, Select(v["arg1_range"], 0) > -1.0e+38), Select(v["arg1_range"], 1) < 1.0e+38))))
)

def rule_15_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 15
        rule_15(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
