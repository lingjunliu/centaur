import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# Given a float is not 0 , and a tensor has a string dtype , then the floating parameter can only be 0 or 1 (Rule 192)

rule_192 = lambda s, v: (
    s.add(If(And((v["arg2_value"] != 0), v["arg1_dtype"] == 11), (Or(v["arg2_value"] == 0, v["arg2_value"] == 1)), True))
)

def rule_192_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 192
        rule_192(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_192(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']})
