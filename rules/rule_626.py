import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if string v_1 is 'ii' or 'ii->i', then tensor v_2 must have at least 1 dimension with shape equals to float number v_3 and it's max element has to be larger than 0 (Rule 626)

rule_626 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_value"] == 0, v["arg1_value"] == 1)), v["arg2_ndim"] > 0), Or([And(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) == v["arg3_value"], Select(v["arg2_range"], 1) > 0)) for i in range(6)]), False)) if n else
          If(And((Or(v["arg1_value"] == 0, v["arg1_value"] == 1)), v["arg2_ndim"] > 0), Or([And(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) == v["arg3_value"], Select(v["arg2_range"], 1) > 0)) for i in range(6)]), False))
)

def rule_626_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not (isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 626
        rule_626(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_626(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
