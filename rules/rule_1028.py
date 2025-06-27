import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If out is provided, output tensor must be of float type (Rule 1028)

rule_1028 = lambda s, v, n=False: (
    s.add(Not(v["arg1_dtype"] == 7) if n else
          v["arg1_dtype"] == 7)
)

def rule_1028_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 1028
        rule_1028(solver, {'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1028(solver, {'arg1_dtype_': arg1['dtype_']}, neg)
