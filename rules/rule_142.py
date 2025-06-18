import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a tensor has integer dtype and also a string variable is given, then the string must be equal to "int" (Rule 142)

rule_142 = lambda s, v: (
    s.add(If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5)), v["arg2_value"] == "int", True))
)

def rule_142_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 142
        rule_142(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_142(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']})
