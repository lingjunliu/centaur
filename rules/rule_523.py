import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if float v_1 is between -1 and 1 and the maximum of tensor v_2 is larger than 5, then minimum of tensor v_2 must be less than -5 (Rule 523)

rule_523 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1), Select(v["arg2_range"], 1) > 5), Select(v["arg2_range"], 0) < -5, False)) if n else
          If(And(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1), Select(v["arg2_range"], 1) > 5), Select(v["arg2_range"], 0) < -5, False))
)

def rule_523_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 523
        rule_523(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_523(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
