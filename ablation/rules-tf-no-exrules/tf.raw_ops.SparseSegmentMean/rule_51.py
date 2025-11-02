import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If sparse_gradient is True and data is float32 or float 64 or half or bfloat16 then the min must be smaller than max (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg2_value"] == True), (Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 6)))), Select(v["arg1_range"], 0) <= Select(v["arg1_range"], 1), True)) if n else
          If(And((v["arg2_value"] == True), (Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 6)))), Select(v["arg1_range"], 0) <= Select(v["arg1_range"], 1), True))
)

def rule_51_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 51
        rule_51(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
