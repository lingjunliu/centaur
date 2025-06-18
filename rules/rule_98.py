import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# A floating point should not be equal to the average value within the tensor (Rule 98)

rule_98 = lambda s, v: (
    s.add(Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, v["arg2_value"] != Select(v["arg1_range"], 0) + Select(v["arg1_range"], 1) / 2)) for i in range(6)]))
)

def rule_98_func(arg1, arg2, solver=None):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 98
        rule_98(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']})
