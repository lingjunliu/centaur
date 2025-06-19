import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the input boolean is true, a tensor's dimension should be greater than or equal to 2, otherwise the dimension should be less than 2 (Rule 208)

rule_208 = lambda s, v: (
    s.add(If(v["arg1_value"], v["arg2_ndim"] >= 2, v["arg2_ndim"] < 2))
)

def rule_208_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 208
        rule_208(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_208(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']})
