import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# seed's elements must be int32 or int64 (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (1 + 1), Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)) for i in range(6)])) if n else
          And([Implies(i < (1 + 1), Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)) for i in range(6)]))
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
