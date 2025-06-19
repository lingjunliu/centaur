import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the bool v_1 is true, then maximum element of tensor v_2 must be greater or equal to int v_3, and data type of v_2 must be greater or equals to the type of integer (Rule 135)

rule_135 = lambda s, v: (
    s.add(If(v["arg1_value"], And(Select(v["arg2_range"], 1) >= v["arg3_value"], v["arg2_dtype"] >= 1), False))
)

def rule_135_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 135
        rule_135(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_135(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']})
