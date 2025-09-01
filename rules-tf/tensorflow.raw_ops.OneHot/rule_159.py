import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If indices values are smaller than 0, then the on_value/off_value needs to have integer data type (Rule 159)

rule_159 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) < 0, And((Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5)), (Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5))), True)) if n else
          If(Select(v["arg1_range"], 0) < 0, And((Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5)), (Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5))), True))
)

def rule_159_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 159
        rule_159(solver, {'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_159(solver, {'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
