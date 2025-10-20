import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If reverse=True and exclusive=True, then the cumprod implementation will fail due to the ScanOp bug if the specified axis value is out of valid range.  (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == True, v["arg3_value"] == True), And(v["arg1_value"] > (0 - v["arg4_ndim"]), v["arg1_value"] < v["arg4_ndim"] - 1), True)) if n else
          If(And(v["arg2_value"] == True, v["arg3_value"] == True), And(v["arg1_value"] > (0 - v["arg4_ndim"]), v["arg1_value"] < v["arg4_ndim"] - 1), True))
)

def rule_102_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 102
        rule_102(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim']}, neg)
