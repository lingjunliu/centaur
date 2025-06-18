import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a bool variable is true, then all tensor values must be true (Rule 55)

rule_55 = lambda s, v: (
    s.add(If(v["arg1_value"] == True, And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_range"], 0) == True, Select(v["arg2_range"], 1) == True)) for i in range(6)]), True))
)

def rule_55_func(arg1, arg2, solver=None):
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

        # Constraints for rule 55
        rule_55(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_range': arg2['range']})
