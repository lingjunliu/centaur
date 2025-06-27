import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# LU_pivots tensor must be contiguous and of int32 dtype. (Rule 1027)

rule_1027 = lambda s, v, n=False: (
    s.add(Not(v["arg1_dtype"] == 3) if n else
          v["arg1_dtype"] == 3)
)

def rule_1027_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 1027
        rule_1027(solver, {'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1027(solver, {'arg1_dtype_': arg1['dtype_']}, neg)
