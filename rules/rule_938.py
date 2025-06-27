import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# Output argument 'out' must be a tensor (Rule 938)

rule_938 = lambda s, v, n=False: (
    s.add(Not(v["arg2_dtype"] == v["arg1_dtype"]) if n else
          v["arg2_dtype"] == v["arg1_dtype"])
)

def rule_938_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 938
        rule_938(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_dtype_': arg2_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_938(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_dtype_': arg2['dtype_']}, neg)
