import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# Given a str v_1, if it is 'sum' or 'mean', then the maximum of the tensor v_2 has to be smaller than a float number 100, otherwise, it has to be larger than 100 (Rule 477)

rule_477 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg1_value"] == 8, v["arg1_value"] == 7)), Select(v["arg2_range"], 1) < 100, Select(v["arg2_range"], 1) > 100)) if n else
          If((Or(v["arg1_value"] == 8, v["arg1_value"] == 7)), Select(v["arg2_range"], 1) < 100, Select(v["arg2_range"], 1) > 100))
)

def rule_477_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 477
        rule_477(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_477(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
