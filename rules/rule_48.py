import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If ndim of tensor v_1 is greater than 0, and if int v_2 is in the range of its shape dimensions, then v_2 must be less than 10 (Rule 48)

rule_48 = lambda s, v: (
    s.add(If(And(And(v["arg1_ndim"] > 0, v["arg2_value"] >= 0), v["arg2_value"] < v["arg1_ndim"]), v["arg2_value"] < 10, False))
)

def rule_48_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 48
        rule_48(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']})
