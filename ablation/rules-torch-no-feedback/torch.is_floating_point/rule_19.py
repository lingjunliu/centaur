import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input should be a Tensor of dtype that is not str or dtype (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] != 11, v["arg1_dtype"] != 12)) if n else
          And(v["arg1_dtype"] != 11, v["arg1_dtype"] != 12))
)

def rule_19_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 19
        rule_19(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_dtype': arg1['dtype']}, neg)
