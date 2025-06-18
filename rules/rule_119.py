import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If String variable is equal to "even" or "odd", then shape[0] % 2 == 0 or shape[0] % 2 == 1, respectively (Rule 119)

rule_119 = lambda s, v: (
    s.add(If(v["arg2_value"] == "even", Select(v["arg1_shape"], 0) * 0.5 == Select(v["arg1_shape"], 0) / 2, If(v["arg2_value"] == "odd", Select(v["arg1_shape"], 0) / 2 != Select(v["arg1_shape"], 0) * 0.5, True)))
)

def rule_119_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 119
        rule_119(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']})
