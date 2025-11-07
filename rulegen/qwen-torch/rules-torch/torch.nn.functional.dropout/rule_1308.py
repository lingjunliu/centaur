import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dropout probability should not be one (Rule 1308)

rule_1308 = lambda s, v, n=False: (
    s.add(Not(v_1 != 1) if n else
          v_1 != 1)
)

def rule_1308_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 1308
        rule_1308(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1308(solver, {}, neg)
