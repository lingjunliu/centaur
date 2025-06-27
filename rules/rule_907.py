import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If string equal sum or mean, maximum value should be greather than 0.01 and minimum value should be less than 0 (Rule 907)

rule_907 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg2_value"] == 8, v["arg2_value"] == 7)), And(Select(v["arg1_range"], 1) > 0.01, Select(v["arg1_range"], 0) < 0), False)) if n else
          If((Or(v["arg2_value"] == 8, v["arg2_value"] == 7)), And(Select(v["arg1_range"], 1) > 0.01, Select(v["arg1_range"], 0) < 0), False))
)

def rule_907_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 907
        rule_907(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_907(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
