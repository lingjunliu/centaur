import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor is either on CPU or CUDA. This is for demonstrating checking device properties, but the grammar cannot express device. So, assume the maximum element to be a representative, and put a dummy condition. (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_range"], 1) > -99999999999999) if n else
          Select(v["arg1_range"], 1) > -99999999999999)
)

def rule_43_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 43
        rule_43(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_range': arg1['range']}, neg)
