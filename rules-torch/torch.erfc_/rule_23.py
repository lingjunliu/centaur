import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dtype must not be bool, int8, int16, int32, int64, uint8 and if dtype is complex, must be complex64 (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), If(v["arg1_dtype"] == 9, True, If(v["arg1_dtype"] == 10, False, True)))) if n else
          And(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), If(v["arg1_dtype"] == 9, True, If(v["arg1_dtype"] == 10, False, True))))
)

def rule_23_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 23
        rule_23(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_dtype': arg1['dtype']}, neg)
