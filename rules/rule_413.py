import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if float v_1 is positive then number of dimensions of tensor v_2 must be within [1,4] (Rule 413)

rule_413 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, And(v["arg2_ndim"] >= 1, v["arg2_ndim"] <= 4), False)) if n else
          If(v["arg1_value"] > 0, And(v["arg2_ndim"] >= 1, v["arg2_ndim"] <= 4), False))
)

def rule_413_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 413
        rule_413(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_413(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
