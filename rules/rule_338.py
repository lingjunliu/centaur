import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If v_1 is tanh, then v_2 should be float (Rule 338)

rule_338 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 11, And(v["arg2_value"] > -1000, v["arg2_value"] < 1000), False)) if n else
          If(v["arg1_value"] == 11, And(v["arg2_value"] > -1000, v["arg2_value"] < 1000), False))
)

def rule_338_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 338
        rule_338(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_338(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
