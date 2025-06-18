import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a boolean variable is true or tensor has dimension 0 then all dtypes are allowed except string (Rule 111)

rule_111 = lambda s, v: (
    s.add(If(Or(v["arg1_value"] == True, v["arg2_ndim"] == 0), v["arg2_dtype"] != 11, True))
)

def rule_111_func(arg1, arg2, solver=None):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 111
        rule_111(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']})
