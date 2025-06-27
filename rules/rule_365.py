import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if string v_1 equals "bn,anm,bm->ba", then shape of v_2 at dimension 0 should be the same as shape of v_2 at dimension 2 (Rule 365)

rule_365 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 5, Select(v["arg2_shape"], 0) == Select(v["arg2_shape"], 2), False)) if n else
          If(v["arg1_value"] == 5, Select(v["arg2_shape"], 0) == Select(v["arg2_shape"], 2), False))
)

def rule_365_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 365
        rule_365(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_365(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)
