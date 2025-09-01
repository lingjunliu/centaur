import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# For numerical dtypes, the operation must not lead to numeric instabilities resulting from excessively large or small value. To do so ensure that the min and max values are between reasonable bounds before and after rounding. (Rule 105)

rule_105 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 11)), And((Select(v["arg1_range"], 0) - (0.5 / (Or(10, v["arg2_value"])))) > -1000, (Select(v["arg1_range"], 1) + (0.5 / (Or(10, v["arg2_value"])))) < 1000), True)) if n else
          If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 11)), And((Select(v["arg1_range"], 0) - (0.5 / (Or(10, v["arg2_value"])))) > -1000, (Select(v["arg1_range"], 1) + (0.5 / (Or(10, v["arg2_value"])))) < 1000), True))
)

def rule_105_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 105
        rule_105(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_105(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
