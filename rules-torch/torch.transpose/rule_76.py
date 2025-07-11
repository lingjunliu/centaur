import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When one dimension is 0 and other is -1 then ndim must be 1, check with abs value (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(Or((And((If(v["arg2_value"] < 0, (0 - v["arg2_value"]), v["arg2_value"])) == 0, (If(v["arg3_value"] < 0, (0 - v["arg3_value"]), v["arg3_value"])) == 1)), (And((If(v["arg2_value"] < 0, (0 - v["arg2_value"]), v["arg2_value"])) == 1, (If(v["arg3_value"] < 0, (0 - v["arg3_value"]), v["arg3_value"])) == 0))), v["arg1_ndim"] == 1, False)) if n else
          If(Or((And((If(v["arg2_value"] < 0, (0 - v["arg2_value"]), v["arg2_value"])) == 0, (If(v["arg3_value"] < 0, (0 - v["arg3_value"]), v["arg3_value"])) == 1)), (And((If(v["arg2_value"] < 0, (0 - v["arg2_value"]), v["arg2_value"])) == 1, (If(v["arg3_value"] < 0, (0 - v["arg3_value"]), v["arg3_value"])) == 0))), v["arg1_ndim"] == 1, False))
)

def rule_76_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 76
        rule_76(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
