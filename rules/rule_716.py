import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if string v_2 equals "tanh", then shape of tensor v_1 in the dimension 1 should be less or equal to 50 (Rule 716)

rule_716 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 11, Select(v["arg1_shape"], 1) <= 50, False)) if n else
          If(v["arg2_value"] == 11, Select(v["arg1_shape"], 1) <= 50, False))
)

def rule_716_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 716
        rule_716(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_716(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
