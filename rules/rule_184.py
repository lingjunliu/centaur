import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if float v_1 > 0.0, then max of tensor v_2 should be greater than 1.0, and also greater or equals than float v_1 (Rule 184)

rule_184 = lambda s, v: (
    s.add(If(v["arg1_value"] > 0.0, And(Select(v["arg2_range"], 1) > 1.0, Select(v["arg2_range"], 1) >= v["arg1_value"]), False))
)

def rule_184_func(arg1, arg2, solver=None):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 184
        rule_184(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_184(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']})
