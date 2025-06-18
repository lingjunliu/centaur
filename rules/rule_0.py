import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# input tensor's data type should be one of the supported types, encoded as an integer (Rule 0)

rule_0 = lambda s, v: (
    s.add(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 1), v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12))
)

def rule_0_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 0
        rule_0(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_0(solver, {'arg2_value': arg2['value']})
