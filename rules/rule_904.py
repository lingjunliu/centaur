import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the input tensor is 3 dimensional, its max, min, and a bool variable equal to true, then max value should be positive and minimum negative. (Rule 904)

rule_904 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_ndim"] == 3), (v["arg2_value"] == True)), And((Select(v["arg1_range"], 1) > 0), (Select(v["arg1_range"], 0) < 0)), False)) if n else
          If(And((v["arg1_ndim"] == 3), (v["arg2_value"] == True)), And((Select(v["arg1_range"], 1) > 0), (Select(v["arg1_range"], 0) < 0)), False))
)

def rule_904_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 904
        rule_904(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_904(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
