import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If tensor is Bool, mean and std must be 0. Otherwise, tensor must be float or complex dtype, std must be a nonnegative finite real number, and the mean must be a finite real number. Moreover the std cannot be NaN (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, And(v["arg2_value"] == 0.0, v["arg3_value"] == 0.0), (And(And(And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (v["arg3_value"] * v["arg3_value"]) >= 0.0), v["arg3_value"] >= 0.0), v["arg3_value"] < 1000000000000000000.0), v["arg3_value"] > -1000000000000000000.0), (v["arg2_value"] * v["arg2_value"]) >= 0.0), v["arg2_value"] < 1000000000000000000.0), v["arg2_value"] > -1000000000000000000.0), v["arg3_value"] == v["arg3_value"])))) if n else
          If(v["arg1_dtype"] == 0, And(v["arg2_value"] == 0.0, v["arg3_value"] == 0.0), (And(And(And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (v["arg3_value"] * v["arg3_value"]) >= 0.0), v["arg3_value"] >= 0.0), v["arg3_value"] < 1000000000000000000.0), v["arg3_value"] > -1000000000000000000.0), (v["arg2_value"] * v["arg2_value"]) >= 0.0), v["arg2_value"] < 1000000000000000000.0), v["arg2_value"] > -1000000000000000000.0), v["arg3_value"] == v["arg3_value"]))))
)

def rule_45_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 45
        rule_45(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
