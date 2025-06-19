import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if the dtype of the tensor v_1 is a boolean and int v_2 is greater than zero, then int v_2 must be in the range of v_1 (0 to 1 (Rule 116)

rule_116 = lambda s, v: (
    s.add(If(v["arg1_dtype"] == 0, And(v["arg2_value"] >= 0, v["arg2_value"] <= 1), False))
)

def rule_116_func(arg1, arg2, solver=None):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 116
        rule_116(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']})
