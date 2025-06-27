import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# Input tensor must be of floating-point type (Rule 959)

rule_959 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)) if n else
          Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8))
)

def rule_959_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 959
        rule_959(solver, {'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_959(solver, {'arg1_dtype_': arg1['dtype_']}, neg)
