import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If string v_2 equals to "tanh", then the number of dimension for the tensor v_1 should be 2 (Rule 691)

rule_691 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 11, v["arg1_ndim"] == 2, False)) if n else
          If(v["arg2_value"] == 11, v["arg1_ndim"] == 2, False))
)

def rule_691_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 691
        rule_691(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_691(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
