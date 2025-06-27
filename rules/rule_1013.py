import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# divisor p should not be equal to zero (Rule 1013)

rule_1013 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] != 0) if n else
          v["arg1_value"] != 0)
)

def rule_1013_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 1013
        rule_1013(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1013(solver, {'arg1_value': arg1['value']}, neg)
