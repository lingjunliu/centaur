import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if bool v_1 is true, and both min and max elements of tensor v_2 are not equal to zero, then max element of v_2 should be greater than 10 (Rule 593)

rule_593 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"], Select(v["arg2_range"], 0) != 0), Select(v["arg2_range"], 1) != 0), v["arg2_ndim"] > 0), Select(v["arg2_range"], 1) > 10, False)) if n else
          If(And(And(And(v["arg1_value"], Select(v["arg2_range"], 0) != 0), Select(v["arg2_range"], 1) != 0), v["arg2_ndim"] > 0), Select(v["arg2_range"], 1) > 10, False))
)

def rule_593_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 593
        rule_593(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_593(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
