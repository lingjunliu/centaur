import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if the product of minimum and maximum element in tensor v_1 is lesser than 0, and string v_2 is not 'none', the the number of dimensions of v_1 must be smaller or equal to 3 (Rule 479)

rule_479 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 0) * Select(v["arg1_range"], 1) < 0, v["arg2_value"] != 6), v["arg1_ndim"] <= 3, False)) if n else
          If(And(Select(v["arg1_range"], 0) * Select(v["arg1_range"], 1) < 0, v["arg2_value"] != 6), v["arg1_ndim"] <= 3, False))
)

def rule_479_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 479
        rule_479(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_479(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
