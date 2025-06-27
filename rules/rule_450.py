import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If v_1 string is equal to "mean", then shape of the last dimension for the tensor v_2 must be an even number (Rule 450)

rule_450 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 7, v["arg2_ndim"] > 0), Or([And(x < (Select(v["arg2_shape"], v["arg2_ndim"] - 1) + 1), x == Select(v["arg2_shape"], v["arg2_ndim"] - 1)) for x in range(6)]), False)) if n else
          If(And(v["arg1_value"] == 7, v["arg2_ndim"] > 0), Or([And(x < (Select(v["arg2_shape"], v["arg2_ndim"] - 1) + 1), x == Select(v["arg2_shape"], v["arg2_ndim"] - 1)) for x in range(6)]), False))
)

def rule_450_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 450
        rule_450(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_450(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
