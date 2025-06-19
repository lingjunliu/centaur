import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If boolean v_1 is true, then tensor v_2 should contain the same value at dimension 0 and dimension 1. (Rule 45)

rule_45 = lambda s, v: (
    s.add(If(v["arg1_value"], Select(v["arg2_shape"], 0) == Select(v["arg2_shape"], 1), False))
)

def rule_45_func(arg1, arg2, solver=None):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']})
