import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the tensor is complex64 or complex128, the minimum value must be greater than -10000 and maximum value must be less than 10000. (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), (And(Select(v["arg1_range"], 0) > -10000, Select(v["arg1_range"], 1) < 10000)), False)) if n else
          If((Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), (And(Select(v["arg1_range"], 0) > -10000, Select(v["arg1_range"], 1) < 10000)), False))
)

def rule_51_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 51
        rule_51(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
