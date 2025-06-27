import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If integer v_1 is less than 10 and the result of the multiplication of min and max values on v_2 tensor is negative, and has a dimension and its shape at index 0 has power of 2, then its dimension must be less than 3 (Rule 598)

rule_598 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"] < 10, (Select(v["arg2_range"], 0) * Select(v["arg2_range"], 1) < 0)), v["arg2_ndim"] > 0), (Or(Or(Or(Select(v["arg2_shape"], 0) == 1, Select(v["arg2_shape"], 0) == 2), Select(v["arg2_shape"], 0) == 4), Select(v["arg2_shape"], 0) == 8))), v["arg2_ndim"] < 3, False)) if n else
          If(And(And(And(v["arg1_value"] < 10, (Select(v["arg2_range"], 0) * Select(v["arg2_range"], 1) < 0)), v["arg2_ndim"] > 0), (Or(Or(Or(Select(v["arg2_shape"], 0) == 1, Select(v["arg2_shape"], 0) == 2), Select(v["arg2_shape"], 0) == 4), Select(v["arg2_shape"], 0) == 8))), v["arg2_ndim"] < 3, False))
)

def rule_598_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 598
        rule_598(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_598(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
