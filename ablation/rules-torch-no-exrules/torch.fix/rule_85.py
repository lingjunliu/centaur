import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Invalid input types are boolean, complex, string, float16, and float32. All dtypes need to be numerical to support fix operation. (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 6), v["arg1_dtype"] != 7), v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), v["arg1_dtype"] != 11)) if n else
          And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 6), v["arg1_dtype"] != 7), v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), v["arg1_dtype"] != 11))
)

def rule_85_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 85
        rule_85(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_dtype': arg1['dtype']}, neg)
