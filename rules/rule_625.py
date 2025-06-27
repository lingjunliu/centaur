import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If v_1 is true and v_2 string has length > 2, then the max of tensor should be greater than 5 (Rule 625)

rule_625 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"], (Select(v["arg2_shape"], 0) > 2)), v["arg3_ndim"] > 0), Select(v["arg3_range"], 1) > 5, False)) if n else
          If(And(And(v["arg1_value"], (Select(v["arg2_shape"], 0) > 2)), v["arg3_ndim"] > 0), Select(v["arg3_range"], 1) > 5, False))
)

def rule_625_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, str)):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg3_ndim == arg3.ndim)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 625
        rule_625(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_625(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim']}, neg)
