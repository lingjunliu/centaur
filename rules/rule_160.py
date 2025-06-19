import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If number v_1 is greater or equals than 1, the max of tensor v_2 is greater than v_1, then string v_3 must be non-empty (Rule 160)

rule_160 = lambda s, v: (
    s.add(If(And(v["arg1_value"] >= 1, Select(v["arg2_range"], 1) > v["arg1_value"]), v["arg3_value"] != "", False))
)

def rule_160_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not (isinstance(arg3, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = String('arg3_value')

        # Value assignments
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 160
        rule_160(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_160(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']})
