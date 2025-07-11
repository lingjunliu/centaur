import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if not complex or not string, the maximum value can be higher. (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(And((And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10)), (v["arg1_dtype"] != 11)), Select(v["arg1_range"], 1) < v["arg2_value"], False)) if n else
          If(And((And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10)), (v["arg1_dtype"] != 11)), Select(v["arg1_range"], 1) < v["arg2_value"], False))
)

def rule_99_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 99
        rule_99(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
