import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the string v_1 is 'min' or 'max', then tensor v_2 must have the same data type (int or float (Rule 118)

rule_118 = lambda s, v: (
    s.add(If(Or(v["arg1_value"] == "min", v["arg1_value"] == "max"), Or((And(And(And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), 1 <= v["arg3_value"]), v["arg3_value"] <= 5)), (And(And(And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), 6 <= v["arg3_value"]), v["arg3_value"] <= 8))), False))
)

def rule_118_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 118
        rule_118(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']})
