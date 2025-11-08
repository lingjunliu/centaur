import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the number is smaller than 0 and greater than -100, and tensor dtype is int then the maximum value should be smaller than 100 (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_value"] < 0), (v["arg1_value"] > -100)), (Or(Or(Or(Or((v["arg2_dtype"] == 1), (v["arg2_dtype"] == 2)), (v["arg2_dtype"] == 3)), (v["arg2_dtype"] == 4)), (v["arg2_dtype"] == 5)))), Select(v["arg2_range"], 1) < 100, True)) if n else
          If(And(And((v["arg1_value"] < 0), (v["arg1_value"] > -100)), (Or(Or(Or(Or((v["arg2_dtype"] == 1), (v["arg2_dtype"] == 2)), (v["arg2_dtype"] == 3)), (v["arg2_dtype"] == 4)), (v["arg2_dtype"] == 5)))), Select(v["arg2_range"], 1) < 100, True))
)

def rule_115_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
