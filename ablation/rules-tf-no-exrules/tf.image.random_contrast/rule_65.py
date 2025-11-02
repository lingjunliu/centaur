import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Comprehensive constraints (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(Or(Or(Or(Or(Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 4)), (v["arg1_dtype"] == 5)), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10))), (v["arg1_ndim"] >= 3)), (v["arg1_dtype"] != 6)), (v["arg3_value"] > v["arg2_value"])), (v["arg2_value"] >= 0))) if n else
          And(And(And(And((Or(Or(Or(Or(Or(Or(Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 4)), (v["arg1_dtype"] == 5)), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10))), (v["arg1_ndim"] >= 3)), (v["arg1_dtype"] != 6)), (v["arg3_value"] > v["arg2_value"])), (v["arg2_value"] >= 0)))
)

def rule_65_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 65
        rule_65(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
