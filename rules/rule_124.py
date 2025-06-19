import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If float v_1 is greater than 0, then the sum of the shape of each dimension of tensor v_2 should be greater than v_1 (Rule 124)

rule_124 = lambda s, v: (
    s.add(If(v["arg1_value"] > 0, Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) + v["arg1_value"] > 0) for i in range(6)]), False))
)

def rule_124_func(arg1, arg2, solver=None):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 124
        rule_124(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']})
