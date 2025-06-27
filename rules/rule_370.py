import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If v_1 string is "tanh", and v_2 tensor's maximum element's absolute value is greater than 1, then v_3 should be set to false (Rule 370)

rule_370 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 11, If((If(Select(v["arg2_range"], 1) < 0, 0 - Select(v["arg2_range"], 1), Select(v["arg2_range"], 1))) > 1, False, v["arg3_value"]), False)) if n else
          If(v["arg1_value"] == 11, If((If(Select(v["arg2_range"], 1) < 0, 0 - Select(v["arg2_range"], 1), Select(v["arg2_range"], 1))) > 1, False, v["arg3_value"]), False))
)

def rule_370_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not (isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 370
        rule_370(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_370(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)
