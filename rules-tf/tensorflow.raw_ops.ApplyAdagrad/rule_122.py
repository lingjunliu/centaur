import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if dtype of var is int, and dtype of grad is not int, then update_slots must be false, and lr must be positive (Rule 122)

rule_122 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(Or(Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 4), v["arg1_dtype"] == 17)), (And(And(And(And(And(v["arg2_dtype"] != 3, v["arg2_dtype"] != 5), v["arg2_dtype"] != 2), v["arg2_dtype"] != 1), v["arg2_dtype"] != 4), v["arg2_dtype"] != 17))), And(v["arg3_value"] == False, Select(v["arg4_range"], 0) > 0), False)) if n else
          If(And((Or(Or(Or(Or(Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 4), v["arg1_dtype"] == 17)), (And(And(And(And(And(v["arg2_dtype"] != 3, v["arg2_dtype"] != 5), v["arg2_dtype"] != 2), v["arg2_dtype"] != 1), v["arg2_dtype"] != 4), v["arg2_dtype"] != 17))), And(v["arg3_value"] == False, Select(v["arg4_range"], 0) > 0), False))
)

def rule_122_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 122
        rule_122(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_122(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_range': arg4['range']}, neg)
