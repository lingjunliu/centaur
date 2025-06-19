import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the number of dimensions of tensor is greater than 2 and less than 5, then the float input should be a valid decimal number between 0 and 1 (Rule 222)

rule_222 = lambda s, v: (
    s.add(If(And(v["arg1_ndim"] > 2, v["arg1_ndim"] < 5), And(v["arg2_value"] > 0, v["arg2_value"] < 1), False))
)

def rule_222_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)

        # Constraints for rule 222
        rule_222(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_222(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']})
