import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the maximum value is less than the minimum, then there will be an inversion and should be fixed (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) < Select(v["arg1_range"], 0), True, True)) if n else
          If(Select(v["arg1_range"], 1) < Select(v["arg1_range"], 0), True, True))
)

def rule_56_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 56
        rule_56(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_range': arg1['range']}, neg)
