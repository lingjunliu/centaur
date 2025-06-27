import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# For a 2D tensor, the sum of all elements must be greater than a given number (Rule 641)

rule_641 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 2, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), Or([And(j < (Select(v["arg1_shape"], 1) - 1 + 1), Select(v["arg1_range"], 0) * Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) > v["arg2_value"]) for j in range(6)])) for i in range(6)]), False)) if n else
          If(v["arg1_ndim"] == 2, Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), Or([And(j < (Select(v["arg1_shape"], 1) - 1 + 1), Select(v["arg1_range"], 0) * Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) > v["arg2_value"]) for j in range(6)])) for i in range(6)]), False))
)

def rule_641_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 641
        rule_641(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_641(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
