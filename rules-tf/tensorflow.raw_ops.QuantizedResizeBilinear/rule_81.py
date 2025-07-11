import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the image data type is float32 then the values of min and max must be in a specific range (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 7, And(Select(v["arg2_range"], 0) > -100, Select(v["arg2_range"], 1) < 100), And(Or(v["arg1_dtype"] != 7, Select(v["arg2_range"], 0) > -100), Select(v["arg2_range"], 1) < 100))) if n else
          If(v["arg1_dtype"] == 7, And(Select(v["arg2_range"], 0) > -100, Select(v["arg2_range"], 1) < 100), And(Or(v["arg1_dtype"] != 7, Select(v["arg2_range"], 0) > -100), Select(v["arg2_range"], 1) < 100)))
)

def rule_81_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 81
        rule_81(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range']}, neg)
