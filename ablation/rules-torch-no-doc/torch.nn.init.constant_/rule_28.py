import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If tensor is float16, value must be able to be represented as a float16 number without loss of precision or range overflow (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), (Or(Or(v["arg2_value"] == 0, v["arg2_value"] > 5.960464477539063e-08), v["arg2_value"] < -5.960464477539063e-08))), True)) if n else
          If(v["arg1_dtype"] == 6, And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), (Or(Or(v["arg2_value"] == 0, v["arg2_value"] > 5.960464477539063e-08), v["arg2_value"] < -5.960464477539063e-08))), True))
)

def rule_28_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 28
        rule_28(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
