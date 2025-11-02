import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if the storage object dtype is equal to 1 or 2, its minimum and maximum values must be smaller than zero (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), And(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) < 0), True)) if n else
          If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), And(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) < 0), True))
)

def rule_102_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 102
        rule_102(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
