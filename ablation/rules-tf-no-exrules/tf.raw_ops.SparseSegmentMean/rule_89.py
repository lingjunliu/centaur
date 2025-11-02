import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the data is float 32, and segment_ids needs to have positive values with sparse_gradient is false, but minimum value exists (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg3_value"] == False), (v["arg1_dtype"] == 7)), (Select(v["arg2_range"], 0) >= 0)), True, True)) if n else
          If(And(And((v["arg3_value"] == False), (v["arg1_dtype"] == 7)), (Select(v["arg2_range"], 0) >= 0)), True, True))
)

def rule_89_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 89
        rule_89(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)
