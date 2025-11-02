import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The `create_graph` and `retain_graph` parameters are only valid for floating-point tensors. (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10)), True, (And(v["arg2_value"] == False, v["arg3_value"] == False)))) if n else
          If(Or(Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10)), True, (And(v["arg2_value"] == False, v["arg3_value"] == False))))
)

def rule_60_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 60
        rule_60(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
