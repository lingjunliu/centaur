import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# Tensor's dtype should be either integer or floating point (Rule 83)

rule_83 = lambda s, v: (
    s.add(Or((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5)), (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8))))
)

def rule_83_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 83
        rule_83(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_dtype': arg1['dtype']})
