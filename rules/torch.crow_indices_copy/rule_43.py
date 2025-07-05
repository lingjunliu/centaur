import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# crow_indices_copy requires input tensor of type integer and the sum of its elements is less than 1000 (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), Select(v["arg1_range"], 1) < 1000)) if n else
          And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), Select(v["arg1_range"], 1) < 1000))
)

def rule_43_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 43
        rule_43(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
