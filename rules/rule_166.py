import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If string v_1 is 'max', then the maximum of tensor v_2 should be smaller or equals than integer v_3 multiplied by float v_4. (Rule 166)

rule_166 = lambda s, v: (
    s.add(If(v["arg1_value"] == "max", Select(v["arg2_range"], 1) <= v["arg3_value"] * v["arg4_value"], False))
)

def rule_166_func(arg1, arg2, arg3, arg4, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False
        if not (isinstance(arg4, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 166
        rule_166(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_166(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']})
