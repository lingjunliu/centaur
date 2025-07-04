import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Input value shouldn't be very close to 0 if the type is float16. (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, Or(Select(v["arg1_range"], 0) < -0.001, Select(v["arg1_range"], 1) > 0.001), False)) if n else
          If(v["arg1_dtype"] == 6, Or(Select(v["arg1_range"], 0) < -0.001, Select(v["arg1_range"], 1) > 0.001), False))
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 23
        rule_23(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
