import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The values inside need to have a limited amount of deviation. To prevent errors downstream from numbers being too high or low for processing or due to noise. (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) < 100000) if n else
          Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) < 100000)
)

def rule_75_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 75
        rule_75(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_range': arg1['range']}, neg)
