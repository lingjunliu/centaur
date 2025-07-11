import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If timeout is set to be non-zero, and min_interval_secs is zero and timeout_fn isn't defined, it will not proceed after timeout (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, v["arg2_value"] == 0), v["arg3_value"] == False, False)) if n else
          If(And(v["arg1_value"] > 0, v["arg2_value"] == 0), v["arg3_value"] == False, False))
)

def rule_64_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 64
        rule_64(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
