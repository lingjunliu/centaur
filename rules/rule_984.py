import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# Input tensor v_1 must be of Double type or Complex types. (Rule 984)

rule_984 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)) if n else
          Or(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))
)

def rule_984_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 984
        rule_984(solver, {'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_984(solver, {'arg1_dtype_': arg1['dtype_']}, neg)
