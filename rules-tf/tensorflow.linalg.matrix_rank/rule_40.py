import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If `tol` is specified and `validate_args` is true, the dtype of 'a' should be float or complex; if `validate_args` is false, the dtype should still be valid (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] > 0, v["arg3_value"] == True), And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), If(v["arg3_value"] == False, And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), True))) if n else
          If(And(v["arg2_value"] > 0, v["arg3_value"] == True), And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), If(v["arg3_value"] == False, And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), True)))
)

def rule_40_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 40
        rule_40(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
