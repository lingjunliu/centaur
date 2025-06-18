import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if a float number is less than the max number and if the max number is less than first element size, the first element has to be greather than integer 1 (Rule 135)

rule_135 = lambda s, v: (
    s.add(If(And(Select(v["arg1_range"], 1) > v["arg2_value"], Select(v["arg1_shape"], 0) > Select(v["arg1_range"], 1)), Select(v["arg1_shape"], 0) > 1, True))
)

def rule_135_func(arg1, arg2, solver=None):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 135
        rule_135(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_135(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']})
