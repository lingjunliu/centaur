import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# When v_1 is false then the dtype of tensor v_2 cannot be float. v_2's number of dimensions is v_3, when inference is false this value v_3 can be at most 2 (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, And(v["arg2_dtype"] != 8, v["arg3_value"] <= 2), False)) if n else
          If(v["arg1_value"] == False, And(v["arg2_dtype"] != 8, v["arg3_value"] <= 2), False))
)

def rule_19_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
