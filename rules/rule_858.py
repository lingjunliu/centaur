import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If string v_2 does not equal to "tanh", and the shape of dimension zero of tensor v_1 is less than 100, then the maximum element of v_1 is greater than or equal to 1 (Rule 858)

rule_858 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] != 11, Select(v["arg1_shape"], 0) < 100), Select(v["arg1_range"], 1) >= 1, False)) if n else
          If(And(v["arg2_value"] != 11, Select(v["arg1_shape"], 0) < 100), Select(v["arg1_range"], 1) >= 1, False))
)

def rule_858_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 858
        rule_858(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_858(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
