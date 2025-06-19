import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If float v_1 is non-negative, then the tensor v_2's dtype must be one of the float types or complex types (Rule 105)

rule_105 = lambda s, v: (
    s.add(If(v["arg1_value"] >= 0, Or((And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8)), (And(9 <= v["arg2_dtype"], v["arg2_dtype"] <= 10))), False))
)

def rule_105_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 105
        rule_105(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_105(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']})
