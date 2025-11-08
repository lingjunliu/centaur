import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The values of lambd and thresh should be sufficiently small relative to the representable range of the tensor datatype. (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, And(v["arg2_value"] < 100, v["arg3_value"] < 100), If(v["arg1_dtype"] == 2, And(v["arg2_value"] < 10000, v["arg3_value"] < 10000), If(v["arg1_dtype"] == 3, And(v["arg2_value"] < 1000000000, v["arg3_value"] < 1000000000), If(v["arg1_dtype"] == 7, And(v["arg2_value"] < 10000, v["arg3_value"] < 10000), True))))) if n else
          If(v["arg1_dtype"] == 1, And(v["arg2_value"] < 100, v["arg3_value"] < 100), If(v["arg1_dtype"] == 2, And(v["arg2_value"] < 10000, v["arg3_value"] < 10000), If(v["arg1_dtype"] == 3, And(v["arg2_value"] < 1000000000, v["arg3_value"] < 1000000000), If(v["arg1_dtype"] == 7, And(v["arg2_value"] < 10000, v["arg3_value"] < 10000), True)))))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 48
        rule_48(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
