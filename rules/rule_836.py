import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if the tensor shape in 0th dimension is 1 and string is tanh then maximum value must be positive and less than 1 (Rule 836)

rule_836 = lambda s, v, n=False: (
    s.add(Not(If(And((Select(v["arg1_shape"], 0) == 1), (v["arg2_value"] == 11)), And((Select(v["arg1_range"], 1) > 0), (Select(v["arg1_range"], 1) < 1)), False)) if n else
          If(And((Select(v["arg1_shape"], 0) == 1), (v["arg2_value"] == 11)), And((Select(v["arg1_range"], 1) > 0), (Select(v["arg1_range"], 1) < 1)), False))
)

def rule_836_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 836
        rule_836(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_836(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
