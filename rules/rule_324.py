import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a boolean is true and the dtype is int, then minimum value must be > 0 (Rule 324)

rule_324 = lambda s, v: (
    s.add(If(And(And(v["arg1_value"] == True, 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) > 0, False))
)

def rule_324_func(arg1, arg2, solver=None):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 324
        rule_324(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_324(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']})
